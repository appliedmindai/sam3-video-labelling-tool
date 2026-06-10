import { useRef, useState, useEffect } from "react";
import { uploadVideo, resumeSession, listSessions, deleteSession, importSession, exportSession } from "../api.ts";
import { evictSession } from "../frameCache.ts";
import type { SessionSummary } from "../types.ts";
import { timeAgo } from "../utils.ts";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload, FolderOpen, Loader2, Trash2, PackageOpen, Download, Clock } from "lucide-react";
import ConfirmDialog from "./ConfirmDialog.tsx";

interface Props {
  onStatus: (msg: string) => void;
  onPipelineStarted: () => void;
}

const RESOLUTION_OPTIONS = [512, 1024, 1440, 2048, 3840];

export default function VideoUpload({ onStatus, onPipelineStarted }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const hasAutoSwitched = useRef(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fps, setFps] = useState(5);
  const [maxResolution, setMaxResolution] = useState(2048);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [resumingId, setResumingId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const importFileRef = useRef<HTMLInputElement>(null);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("new");

  const busy = uploading || importing || resumingId != null;

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch(() => {});
  }, []);

  // Auto-switch to "recent" tab when sessions load for the first time
  useEffect(() => {
    if (!hasAutoSwitched.current && sessions.length > 0 && activeTab === "new") {
      hasAutoSwitched.current = true;
      setActiveTab("recent");
    }
  }, [sessions]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    onStatus("Uploading video...");
    try {
      await uploadVideo(file, fps, (fraction) => {
        setUploadProgress(fraction);
      }, maxResolution);
      onPipelineStarted();
    } catch (err) {
      onStatus(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }

  async function handleResume(session: SessionSummary) {
    if (busy) return;
    setResumingId(session.session_id);
    onStatus(`Loading "${session.original_name}"...`);
    try {
      await resumeSession(session.session_id);
      onPipelineStarted();
    } catch (err) {
      onStatus(`Resume failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setResumingId(null);
    }
  }

  function handleDeleteClick(e: React.MouseEvent, session: SessionSummary) {
    e.stopPropagation();
    setPendingDelete(session);
  }

  async function handleDeleteConfirm() {
    if (!pendingDelete || deletingId) return;
    const session = pendingDelete;
    setPendingDelete(null);
    setDeletingId(session.session_id);
    try {
      await deleteSession(session.session_id);
      evictSession(session.session_id).catch(() => {}); // fire-and-forget cleanup
      setSessions((prev) => prev.filter((s) => s.session_id !== session.session_id));
    } catch (err) {
      onStatus(`Delete error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDeletingId(null);
    }
  }

  async function handleExport(e: React.MouseEvent, session: SessionSummary) {
    e.stopPropagation();
    if (exportingId) return;
    setExportingId(session.session_id);
    try {
      const blob = await exportSession(session.session_id, false);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const baseName = session.original_name.replace(/\.[^.]+$/, "");
      a.download = `${baseName}_session.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      onStatus(`Export error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExportingId(null);
    }
  }

  function handleFileChange() {
    const file = fileRef.current?.files?.[0];
    setSelectedFile(file ? file.name : null);
  }

  async function handleImport() {
    if (!importFile) return;
    setImporting(true);
    setUploadProgress(0);
    onStatus("Importing session...");

    try {
      const result = await importSession(importFile, (fraction) => {
        setUploadProgress(fraction);
      });
      // Trigger the pipeline (model init) for the imported session
      await resumeSession(result.session_id);
      onPipelineStarted();
    } catch (err) {
      onStatus(`Import error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setImporting(false);
      setUploadProgress(0);
      setImportFile(null);
      if (importFileRef.current) importFileRef.current.value = "";
    }
  }

  // Merge live + previous sessions into a single list, sorted by updated_at desc
  const sortedSessions = [...sessions].sort(
    (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  );

  return (
    <div className="mx-auto flex h-[28rem] w-[40rem] flex-col gap-6">
      <div className="flex shrink-0 flex-col items-center gap-3 pt-4">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          SAM3 Video Labelling Tool
        </h1>
        <p className="text-sm text-muted-foreground">
          Video Mask Segmentation &amp; Annotation
        </p>
      </div>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="grid h-11 w-full shrink-0 grid-cols-3">
          <TabsTrigger value="recent" className="gap-1.5 text-[13px] font-semibold">
            <Clock className="h-4 w-4" />
            Recent
          </TabsTrigger>
          <TabsTrigger value="new" className="gap-1.5 text-[13px] font-semibold">
            <Upload className="h-4 w-4" />
            New Video
          </TabsTrigger>
          <TabsTrigger value="import" className="gap-1.5 text-[13px] font-semibold">
            <PackageOpen className="h-4 w-4" />
            Import
          </TabsTrigger>
        </TabsList>

        {/* Recent — merged session list */}
        <TabsContent value="recent" className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-2 pt-2">
            {sortedSessions.length === 0 ? (
              <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border bg-muted/50 px-4 py-10 text-center">
                <p className="text-sm font-medium text-muted-foreground">No sessions yet</p>
                <p className="text-xs text-muted-foreground">
                  Upload a video or import a session to get started
                </p>
              </div>
            ) : (
              sortedSessions.map((s) => (
                <div
                  key={s.session_id}
                  className={`group flex items-start gap-3 rounded-lg border bg-card px-4 py-3 transition-colors hover:border-primary/30 hover:bg-accent/30 ${
                    s.live
                      ? "border-border border-l-2 border-l-emerald-500/60 hover:border-l-emerald-500/60"
                      : "border-border"
                  }`}
                >
                  <button
                    onClick={() => handleResume(s)}
                    disabled={busy}
                    className="flex min-w-0 flex-1 flex-col gap-1 text-left"
                  >
                    <div className="flex items-center gap-2">
                      <FolderOpen className="h-4 w-4 shrink-0 text-muted-foreground group-hover:text-primary" />
                      <p className="text-sm font-medium text-foreground">
                        {s.original_name}
                      </p>
                      {s.live && (
                        <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-600">
                          Live
                        </span>
                      )}
                      {resumingId === s.session_id && (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {s.frame_count} frames &middot; {s.class_count} classes &middot; {s.object_count} objects
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatBytes(s.disk_size)} &middot; Edited {timeAgo(s.updated_at)}
                    </p>
                  </button>
                  <div className="mt-0.5 flex shrink-0 gap-0.5">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100"
                      onClick={(e) => handleExport(e, s)}
                      disabled={exportingId === s.session_id}
                      title="Export session bundle"
                    >
                      {exportingId === s.session_id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                      ) : (
                        <Download className="h-3.5 w-3.5 text-muted-foreground hover:text-primary" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100"
                      onClick={(e) => handleDeleteClick(e, s)}
                      disabled={deletingId === s.session_id}
                      title="Delete session"
                    >
                      {deletingId === s.session_id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive" />
                      )}
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </TabsContent>

        {/* New Video — file picker + settings + upload */}
        <TabsContent value="new" className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-4 pt-2">
            <div
              className="group relative flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-border bg-muted/50 px-4 py-8 transition-colors hover:border-primary/40 hover:bg-accent/30"
              onClick={() => !busy && fileRef.current?.click()}
            >
              <Upload className="h-6 w-6 text-muted-foreground group-hover:text-primary" />
              <span className="text-sm text-muted-foreground group-hover:text-foreground">
                {selectedFile ?? "Choose video file"}
              </span>
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                disabled={busy}
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            {/* Settings row */}
            <div className="grid grid-cols-2 gap-4">
              {/* Max Sampling FPS */}
              <div className="flex flex-col gap-2">
                <div className="flex items-baseline justify-between">
                  <Label className="text-xs font-medium text-foreground">Max Sampling FPS</Label>
                  <span className="text-xs tabular-nums text-muted-foreground">{fps}</span>
                </div>
                <Slider
                  value={[fps]}
                  min={5}
                  max={60}
                  step={1}
                  onValueChange={([v]) => setFps(v)}
                  disabled={busy}
                />
                <p className="text-[11px] leading-tight text-muted-foreground">
                  Sampling rate used to turn video into frames
                </p>
              </div>

              {/* Max Resolution */}
              <div className="flex flex-col gap-2">
                <div className="flex items-baseline justify-between">
                  <Label className="text-xs font-medium text-foreground">Max Resolution</Label>
                  <span className="text-xs tabular-nums text-muted-foreground">{maxResolution}px</span>
                </div>
                <Slider
                  value={[RESOLUTION_OPTIONS.indexOf(maxResolution)]}
                  min={0}
                  max={RESOLUTION_OPTIONS.length - 1}
                  step={1}
                  onValueChange={([v]) => setMaxResolution(RESOLUTION_OPTIONS[v])}
                  disabled={busy}
                />
                <p className="text-[11px] leading-tight text-muted-foreground">
                  Longest side is scaled down to this size
                </p>
              </div>
            </div>

            <Button
              onClick={handleUpload}
              disabled={busy || !selectedFile}
              size="sm"
              className="h-9 gap-1.5 text-sm"
            >
              {uploading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {uploading ? `${Math.round(uploadProgress * 100)}%` : "Upload"}
            </Button>
          </div>
        </TabsContent>

        {/* Import — zip file picker + import button */}
        <TabsContent value="import" className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-4 pt-2">
            <div
              className="group relative flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-border bg-muted/50 px-4 py-6 transition-colors hover:border-primary/40 hover:bg-accent/30"
              onClick={() => !busy && importFileRef.current?.click()}
            >
              <PackageOpen className="h-6 w-6 text-muted-foreground group-hover:text-primary" />
              <span className="text-sm text-muted-foreground group-hover:text-foreground">
                {importFile ? importFile.name : "Choose session bundle (.zip)"}
              </span>
              <input
                ref={importFileRef}
                type="file"
                accept=".zip"
                disabled={busy}
                onChange={() => {
                  const file = importFileRef.current?.files?.[0] ?? null;
                  setImportFile(file);
                }}
                className="hidden"
              />
            </div>

            <Button
              onClick={handleImport}
              disabled={busy || !importFile}
              variant="outline"
              size="sm"
              className="h-9 gap-1.5 text-sm"
            >
              {importing && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {importing ? `Importing... ${Math.round(uploadProgress * 100)}%` : "Import"}
            </Button>
          </div>
        </TabsContent>
      </Tabs>

      <ConfirmDialog
        open={pendingDelete != null}
        title="Delete Session"
        description={`Delete "${pendingDelete?.original_name}"? All annotations and extracted frames will be permanently removed.`}
        confirmLabel="Delete"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
