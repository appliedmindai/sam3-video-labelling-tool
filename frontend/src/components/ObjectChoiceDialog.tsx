import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { TrackedObject } from "../types.ts";

interface Props {
  open: boolean;
  className: string;
  existingObjects: TrackedObject[];
  onChoose: (objId: number | "new") => void;
  onCancel: () => void;
}

export default function ObjectChoiceDialog({
  open,
  className: clsName,
  existingObjects,
  onChoose,
  onCancel,
}: Props) {
  return (
    <Dialog open={open}>
      <DialogContent
        showCloseButton={false}
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Choose Object</DialogTitle>
          <DialogDescription>
            Class "{clsName}" already has {existingObjects.length === 1 ? "an object" : `${existingObjects.length} objects`}. Use an existing object or create a new one?
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-1.5 py-2">
          {existingObjects.map((obj) => (
            <Button
              key={obj.obj_id}
              variant="outline"
              size="sm"
              className="justify-start text-xs"
              onClick={() => onChoose(obj.obj_id)}
            >
              {clsName} #{obj.obj_id}
            </Button>
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="default" onClick={() => onChoose("new")}>
            Create New
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
