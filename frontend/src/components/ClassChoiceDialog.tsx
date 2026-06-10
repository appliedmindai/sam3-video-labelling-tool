import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ObjectClass } from "../types.ts";

interface Props {
  open: boolean;
  objId: number;
  classes: ObjectClass[];
  onChooseClass: (classId: number) => void;
  onCreateClass: (name: string) => void;
  onCancel: () => void;
}

export default function ClassChoiceDialog({
  open,
  objId,
  classes,
  onChooseClass,
  onCreateClass,
  onCancel,
}: Props) {
  const [newClassName, setNewClassName] = useState("");

  return (
    <Dialog open={open}>
      <DialogContent
        showCloseButton={false}
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Assign Class</DialogTitle>
          <DialogDescription>
            Object #{objId} has no class. Pick an existing class or create a new
            one.
          </DialogDescription>
        </DialogHeader>
        {classes.length > 0 && (
          <div className="flex flex-col gap-1.5 py-2">
            {classes.map((cls) => (
              <Button
                key={cls.id}
                variant="outline"
                size="sm"
                className="justify-start gap-2 text-xs"
                onClick={() => onChooseClass(cls.id)}
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: cls.color }}
                />
                {cls.name}
              </Button>
            ))}
          </div>
        )}
        <div className="flex gap-1.5">
          <Input
            type="text"
            placeholder="New class name"
            value={newClassName}
            onChange={(e) => setNewClassName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && newClassName.trim()) {
                onCreateClass(newClassName.trim());
                setNewClassName("");
              }
            }}
            className="h-8 text-xs"
          />
          <Button
            variant="default"
            size="sm"
            className="h-8 shrink-0"
            disabled={!newClassName.trim()}
            onClick={() => {
              onCreateClass(newClassName.trim());
              setNewClassName("");
            }}
          >
            Create
          </Button>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
