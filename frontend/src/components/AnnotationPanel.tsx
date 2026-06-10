import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import type { ObjectClass, TrackedObject, ConfidenceWarning } from "../types";
import { createClass, deleteClass, renameClass, updateClassColor, reassignObject } from "../api";
import ConfirmDialog from "./ConfirmDialog";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { Plus, X, Eye, EyeOff, ChevronRight, Download, Upload } from "lucide-react";

export const PALETTE = [
  "#6366f1", "#ec4899", "#f59e0b", "#10b981", "#3b82f6",
  "#8b5cf6", "#14b8a6", "#f43f5e", "#84cc16", "#f97316",
  "#06b6d4", "#d946ef", "#a855f7", "#eab308", "#ef4444",
];

interface Props {
  classes: ObjectClass[];
  objects: TrackedObject[];
  selectedClassId: number | null;
  selectedObjId: number | null;
  sessionId: string | null;
  hiddenClassIds: Set<number>;
  frameObjIndex: Map<number, Set<number>>;
  frameCount: number;
  confidenceWarnings: ConfidenceWarning[];
  onClassCreated: (cls: ObjectClass) => void;
  onClassDeleted: (classId: number) => void;
  onClassSelected: (classId: number | null) => void;
  onClassRenamed?: (classId: number, newName: string) => void;
  onClassColorChanged?: (classId: number, color: string) => void;
  onToggleVisibility: (classId: number) => void;
  onObjectSelected: (objId: number | null) => void;
  onObjectDeleted: (objId: number) => void;
  onObjectReassigned: (objId: number, newClassId: number) => void;
  selectedObjIds: Set<number>;
  onToggleObjSelection: (objId: number) => void;
}

export default function AnnotationPanel({
  classes,
  objects,
  selectedClassId,
  selectedObjId,
  sessionId,
  hiddenClassIds,
  frameObjIndex,
  frameCount,
  confidenceWarnings,
  onClassCreated,
  onClassDeleted,
  onClassSelected,
  onClassRenamed,
  onClassColorChanged,
  onToggleVisibility,
  onObjectSelected,
  onObjectDeleted,
  onObjectReassigned,
  selectedObjIds,
  onToggleObjSelection,
}: Props) {
  const [collapsedClassIds, setCollapsedClassIds] = useState<Set<number>>(new Set());
  const [creating, setCreating] = useState(false);
  const [newClassName, setNewClassName] = useState("");
  const [editingClassId, setEditingClassId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [colorPickerClassId, setColorPickerClassId] = useState<number | null>(null);
  const [pendingDeleteClassId, setPendingDeleteClassId] = useState<number | null>(null);
  const [pendingDeleteObjId, setPendingDeleteObjId] = useState<number | null>(null);
  const [focusedIdx, setFocusedIdx] = useState<number>(-1);

  const createInputRef = useRef<HTMLInputElement>(null);
  const editInputRef = useRef<HTMLInputElement>(null);

  // Coverage: per-object count of frames with masks
  const objCoverage = useMemo(() => {
    const map = new Map<number, number>();
    for (const [, objIds] of frameObjIndex) {
      for (const objId of objIds) {
        map.set(objId, (map.get(objId) ?? 0) + 1);
      }
    }
    return map;
  }, [frameObjIndex]);

  // Confidence: set of obj_ids that have warnings
  const objHasWarning = useMemo(
    () => new Set(confidenceWarnings.map((w) => w.obj_id)),
    [confidenceWarnings],
  );

  // Flat list for keyboard navigation + index lookup
  const flatItems = useMemo(() => {
    const items: Array<{ type: 'class'; classId: number } | { type: 'object'; objId: number; classId: number }> = [];
    for (const cls of classes) {
      items.push({ type: 'class', classId: cls.id });
      if (!collapsedClassIds.has(cls.id) && !hiddenClassIds.has(cls.id)) {
        for (const obj of objects.filter(o => o.class_id === cls.id)) {
          items.push({ type: 'object', objId: obj.obj_id, classId: cls.id });
        }
      }
    }
    return items;
  }, [classes, objects, collapsedClassIds, hiddenClassIds]);

  const flatItemIndex = useMemo(() => {
    const map = new Map<string, number>();
    flatItems.forEach((item, idx) => {
      const key = item.type === 'class' ? `class-${item.classId}` : `obj-${item.objId}`;
      map.set(key, idx);
    });
    return map;
  }, [flatItems]);

  useEffect(() => {
    if (editingClassId != null) {
      editInputRef.current?.focus();
      editInputRef.current?.select();
    }
  }, [editingClassId]);

  useEffect(() => {
    if (creating) {
      createInputRef.current?.focus();
    }
  }, [creating]);

  // --- Import / Export classes ---

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExportClasses = useCallback(() => {
    if (classes.length === 0) return;
    const payload = classes.map(({ name, color }) => ({ name, color }));
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "classes.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [classes]);

  const handleImportClasses = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !sessionId) return;
    try {
      const text = await file.text();
      const imported: { name: string; color: string }[] = JSON.parse(text);
      if (!Array.isArray(imported)) throw new Error("Expected an array");

      const existingNames = new Set(classes.map(c => c.name.trim().toLowerCase()));
      let added = 0;
      for (const entry of imported) {
        const name = entry.name?.trim();
        if (!name || !entry.color) continue;
        if (existingNames.has(name.toLowerCase())) continue;
        const cls = await createClass(sessionId, name, entry.color);
        onClassCreated(cls);
        existingNames.add(name.toLowerCase());
        added++;
      }
      if (added === 0) {
        // All classes already existed — no-op
      }
    } catch (err) {
      console.error("Failed to import classes:", err);
    }
    // Reset the file input so the same file can be re-imported
    e.target.value = "";
  }, [sessionId, classes, onClassCreated]);

  // --- Class management handlers ---

  const handleCreate = useCallback(async () => {
    if (!newClassName.trim() || !sessionId) return;
    try {
      const color = PALETTE[classes.length % PALETTE.length];
      const cls = await createClass(sessionId, newClassName.trim(), color);
      onClassCreated(cls);
      setNewClassName("");
      setCreating(false);
    } catch (err) {
      console.error("Failed to create class:", err);
    }
  }, [newClassName, sessionId, classes.length, onClassCreated]);

  const handleDeleteClass = useCallback(async (classId: number) => {
    if (!sessionId) return;
    try {
      await deleteClass(sessionId, classId);
      onClassDeleted(classId);
    } catch (err) {
      console.error("Failed to delete class:", err);
    }
  }, [sessionId, onClassDeleted]);

  function startEditing(cls: ObjectClass) {
    setEditingClassId(cls.id);
    setEditValue(cls.name);
  }

  const commitRename = useCallback(async () => {
    if (editingClassId == null || !sessionId) return;
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== classes.find((c) => c.id === editingClassId)?.name) {
      try {
        await renameClass(sessionId, editingClassId, trimmed);
        onClassRenamed?.(editingClassId, trimmed);
      } catch (err) {
        console.error("Failed to rename class:", err);
      }
    }
    setEditingClassId(null);
  }, [editingClassId, editValue, sessionId, classes, onClassRenamed]);

  function cancelEditing() {
    setEditingClassId(null);
  }

  function toggleCollapse(classId: number) {
    setCollapsedClassIds(prev => {
      const next = new Set(prev);
      if (next.has(classId)) {
        next.delete(classId);
      } else {
        next.add(classId);
      }
      return next;
    });
  }

  const handleReassign = useCallback(async (objId: number, newClassId: number) => {
    if (!sessionId) return;
    try {
      await reassignObject(sessionId, objId, newClassId);
      onObjectReassigned(objId, newClassId);
    } catch (err) {
      console.error("Failed to reassign object:", err);
    }
  }, [sessionId, onObjectReassigned]);

  // --- Keyboard navigation ---

  function handlePanelKeyDown(e: React.KeyboardEvent) {
    if (e.target instanceof HTMLInputElement) return;
    if (focusedIdx < 0 && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      setFocusedIdx(0);
      e.preventDefault();
      return;
    }
    const item = flatItems[focusedIdx];
    if (!item) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setFocusedIdx(Math.min(focusedIdx + 1, flatItems.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setFocusedIdx(Math.max(focusedIdx - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (item.type === 'class') onClassSelected(selectedClassId === item.classId ? null : item.classId);
        else onObjectSelected(selectedObjId === item.objId ? null : item.objId);
        break;
      case 'Delete':
      case 'Backspace':
        e.preventDefault();
        if (item.type === 'class') setPendingDeleteClassId(item.classId);
        else setPendingDeleteObjId(item.objId);
        break;
      case 'ArrowRight':
        e.preventDefault();
        if (item.type === 'class') {
          setCollapsedClassIds(prev => { const next = new Set(prev); next.delete(item.classId); return next; });
        }
        break;
      case 'ArrowLeft':
        e.preventDefault();
        if (item.type === 'class') {
          setCollapsedClassIds(prev => new Set(prev).add(item.classId));
        }
        break;
    }
  }

  // --- Render helpers ---

  // (flatItemIndex map used for keyboard focus ring)

  return (
    <>
      <div
        className="flex flex-col gap-3"
        tabIndex={0}
        onKeyDown={handlePanelKeyDown}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Annotations
          </h3>
          <div className="flex items-center gap-1">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!sessionId}
              title="Import classes"
              className="rounded p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
            >
              <Upload className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={handleExportClasses}
              disabled={classes.length === 0}
              title="Export classes"
              className="rounded p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImportClasses}
              className="hidden"
            />
          </div>
        </div>

        {classes.length === 0 && !creating ? (
          // Empty state
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <p className="text-xs text-muted-foreground">
              Start by creating a class, then click on the video to annotate your first object.
            </p>
            <button
              className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              onClick={() => setCreating(true)}
              disabled={!sessionId}
            >
              <Plus className="h-3.5 w-3.5" />
              Create first class
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-0.5">
            {[...classes].sort((a, b) => a.name.localeCompare(b.name)).map((cls) => {
              const classObjects = objects.filter(o => o.class_id === cls.id);
              const expanded = !collapsedClassIds.has(cls.id);
              const classItemIdx = flatItemIndex.get(`class-${cls.id}`) ?? -1;

              return (
                <div key={cls.id}>
                  {/* Class row */}
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => onClassSelected(selectedClassId === cls.id ? null : cls.id)}
                    className={`group flex cursor-pointer items-center gap-1.5 rounded-md px-1.5 py-1.5 text-left transition-colors ${
                      selectedClassId === cls.id
                        ? "bg-accent ring-1 ring-primary/30"
                        : "hover:bg-accent"
                    } ${focusedIdx === classItemIdx ? "outline outline-2 outline-primary/50" : ""}`}
                    style={selectedClassId === cls.id ? { borderLeft: `2px solid ${cls.color}` } : { borderLeft: '2px solid transparent' }}
                  >
                    {/* Color dot with popover */}
                    <Popover
                      open={colorPickerClassId === cls.id}
                      onOpenChange={(open) => setColorPickerClassId(open ? cls.id : null)}
                    >
                      <PopoverTrigger asChild>
                        <button
                          className="h-2.5 w-2.5 shrink-0 rounded-full ring-offset-background transition-transform hover:scale-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                          style={{ backgroundColor: cls.color }}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-2" align="start" side="bottom">
                        <div className="grid grid-cols-5 gap-1">
                          {PALETTE.map((color) => (
                            <button
                              key={color}
                              className={`h-6 w-6 rounded-full transition-transform hover:scale-110 ${
                                color === cls.color ? "ring-2 ring-primary ring-offset-2" : ""
                              }`}
                              style={{ backgroundColor: color }}
                              onClick={async (e) => {
                                e.stopPropagation();
                                if (color === cls.color) { setColorPickerClassId(null); return; }
                                try {
                                  await updateClassColor(sessionId!, cls.id, color);
                                  onClassColorChanged?.(cls.id, color);
                                } catch (err) {
                                  console.error("Failed to update class color:", err);
                                }
                                setColorPickerClassId(null);
                              }}
                            />
                          ))}
                        </div>
                      </PopoverContent>
                    </Popover>

                    {/* Class name (double-click to rename) */}
                    {editingClassId === cls.id ? (
                      <input
                        ref={editInputRef}
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onBlur={commitRename}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") { e.preventDefault(); commitRename(); }
                          if (e.key === "Escape") { e.preventDefault(); cancelEditing(); }
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="flex-1 rounded border border-primary/30 bg-card px-1 py-0 text-xs font-medium outline-none focus:ring-1 focus:ring-primary/40"
                      />
                    ) : (
                      <span
                        className="flex-1 truncate text-xs font-medium"
                        onDoubleClick={(e) => {
                          e.stopPropagation();
                          startEditing(cls);
                        }}
                      >
                        {cls.name}
                      </span>
                    )}

                    {/* Object count badge */}
                    <span className="text-[10px] tabular-nums text-muted-foreground">{classObjects.length}</span>

                    {/* Visibility toggle */}
                    <button
                      className="rounded p-0.5 text-muted-foreground/50 transition-colors hover:text-foreground"
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleVisibility(cls.id);
                      }}
                    >
                      {hiddenClassIds.has(cls.id) ? (
                        <EyeOff className="h-3 w-3" />
                      ) : (
                        <Eye className="h-3 w-3" />
                      )}
                    </button>

                    {/* Delete button */}
                    <button
                      className="rounded p-0.5 text-muted-foreground/50 transition-colors hover:bg-destructive/10 hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPendingDeleteClassId(cls.id);
                      }}
                    >
                      <X className="h-3 w-3" />
                    </button>

                    {/* Chevron */}
                    <button
                      className="rounded p-0.5 text-muted-foreground transition-transform hover:text-foreground"
                      style={{ transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                      onClick={(e) => { e.stopPropagation(); toggleCollapse(cls.id); }}
                    >
                      <ChevronRight className="h-3 w-3" />
                    </button>
                  </div>

                  {/* Object rows (when expanded and class not hidden) */}
                  {expanded && !hiddenClassIds.has(cls.id) && classObjects.map((obj) => {
                    const objItemIdx = flatItemIndex.get(`obj-${obj.obj_id}`) ?? -1;
                    const coverage = objCoverage.get(obj.obj_id) ?? 0;
                    const hasWarning = objHasWarning.has(obj.obj_id);

                    return (
                      <ContextMenu key={obj.obj_id}>
                        <ContextMenuTrigger asChild>
                          <button
                            onClick={(e) => {
                              if (e.shiftKey) {
                                onToggleObjSelection(obj.obj_id);
                              } else {
                                onObjectSelected(selectedObjId === obj.obj_id ? null : obj.obj_id);
                              }
                            }}
                            className={`group flex w-full items-center gap-2 rounded-md pl-5 pr-2 py-1.5 text-left transition-colors ${
                              selectedObjIds.has(obj.obj_id)
                                ? "bg-accent ring-1 ring-primary/30"
                                : "hover:bg-accent"
                            } ${focusedIdx === objItemIdx ? "outline outline-2 outline-primary/50" : ""}`}
                            style={selectedObjIds.has(obj.obj_id) ? { borderLeft: `2px solid ${cls.color}` } : { borderLeft: '2px solid transparent' }}
                          >
                            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: cls.color }} />
                            <span className="flex-1 truncate text-xs font-medium">
                              Object <span className="font-normal text-muted-foreground">#{obj.obj_id}</span>
                            </span>
                            {/* Coverage bar */}
                            <div className="h-1 w-10 shrink-0 rounded-full bg-muted">
                              <div
                                className="h-full rounded-full"
                                style={{
                                  backgroundColor: cls.color,
                                  width: `${frameCount > 0 ? Math.min(100, (coverage / frameCount) * 100) : 0}%`,
                                  opacity: 0.7,
                                }}
                              />
                            </div>
                            {/* Confidence dot */}
                            {coverage > 0 && (
                              <span
                                className="h-1.5 w-1.5 shrink-0 rounded-full"
                                style={{ backgroundColor: hasWarning ? '#f59e0b' : '#10b981' }}
                              />
                            )}
                            {/* Delete button */}
                            <button
                              className="rounded p-0.5 text-muted-foreground/50 transition-colors hover:bg-destructive/10 hover:text-destructive"
                              onClick={(e) => { e.stopPropagation(); setPendingDeleteObjId(obj.obj_id); }}
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </button>
                        </ContextMenuTrigger>
                        <ContextMenuContent>
                          {classes.length > 1 && (
                            <ContextMenuSub>
                              <ContextMenuSubTrigger>Move to...</ContextMenuSubTrigger>
                              <ContextMenuSubContent>
                                {classes.filter(c => c.id !== cls.id).map(targetCls => (
                                  <ContextMenuItem
                                    key={targetCls.id}
                                    onClick={() => handleReassign(obj.obj_id, targetCls.id)}
                                  >
                                    <span className="mr-2 h-2 w-2 rounded-full" style={{ backgroundColor: targetCls.color }} />
                                    {targetCls.name}
                                  </ContextMenuItem>
                                ))}
                              </ContextMenuSubContent>
                            </ContextMenuSub>
                          )}
                        </ContextMenuContent>
                      </ContextMenu>
                    );
                  })}
                </div>
              );
            })}

            {/* Ghost row / class creation */}
            {creating ? (
              <div className="flex items-center gap-2 px-2 py-1">
                <Plus className="h-3.5 w-3.5 text-muted-foreground" />
                <input
                  ref={createInputRef}
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") { e.preventDefault(); handleCreate(); }
                    if (e.key === "Escape") { e.preventDefault(); setCreating(false); setNewClassName(""); }
                  }}
                  onBlur={() => { if (!newClassName.trim()) { setCreating(false); setNewClassName(""); } }}
                  placeholder="Class name"
                  className="flex-1 rounded border border-primary/30 bg-card px-1 py-0 text-xs font-medium outline-none focus:ring-1 focus:ring-primary/40"
                  autoFocus
                />
              </div>
            ) : (
              <button
                className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                onClick={() => setCreating(true)}
                disabled={!sessionId}
              >
                <Plus className="h-3.5 w-3.5" />
                New class...
              </button>
            )}
          </div>
        )}
      </div>

      {/* Class delete confirmation */}
      {(() => {
        const pendingClass = classes.find(c => c.id === pendingDeleteClassId);
        const affectedCount = objects.filter(o => o.class_id === pendingDeleteClassId).length;
        return (
          <ConfirmDialog
            open={pendingDeleteClassId != null}
            title={`Delete class "${pendingClass?.name}"?`}
            description={
              affectedCount > 0
                ? `This will permanently delete ${affectedCount} object${affectedCount > 1 ? "s" : ""} and all their masks and prompts. This cannot be undone.`
                : "This class has no objects. This cannot be undone."
            }
            confirmLabel="Delete"
            onConfirm={() => { if (pendingDeleteClassId != null) handleDeleteClass(pendingDeleteClassId); setPendingDeleteClassId(null); }}
            onCancel={() => setPendingDeleteClassId(null)}
          />
        );
      })()}

      {/* Object delete confirmation */}
      <ConfirmDialog
        open={pendingDeleteObjId != null}
        title={`Delete object #${pendingDeleteObjId}?`}
        description="This will permanently delete all masks and prompts for this object. This cannot be undone."
        confirmLabel="Delete"
        onConfirm={() => { if (pendingDeleteObjId != null) onObjectDeleted(pendingDeleteObjId); setPendingDeleteObjId(null); }}
        onCancel={() => setPendingDeleteObjId(null)}
      />
    </>
  );
}
