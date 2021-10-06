from event.arguments.implicit_arg_resources import ImplicitArgResources


class FrameSlots:
    def __init__(self, frame_formalism: str, resources: ImplicitArgResources):
        self.frame_formalism = frame_formalism

        self.frame_dep_map = resources.h_frame_dep_map
        self.nom_dep_map = resources.h_nom_dep_map

        if self.frame_formalism == "FrameNet":
            self.frame_slots = resources.h_frame_slots
        elif self.frame_formalism == "Propbank":
            self.frame_slots = resources.h_nom_slots
        else:
            raise ValueError("Unknown frame formalism.")

    def get_dep_from_slot(self, event, slot):
        """Given the predicate and a slot, we compute the dependency used to
        create a dep-word.

        Args:
          event:
          slot:

        Returns:

        """
        dep = "unk_dep"

        if self.frame_formalism == "FrameNet":
            fid = event["frame"]
            pred = event["predicate"]
            if not fid == -1:
                if (fid, slot, pred) in self.frame_dep_map:
                    dep = self.frame_dep_map[(fid, slot, pred)]
        elif self.frame_formalism == "Propbank":
            pred = event["predicate"]
            if (pred, slot) in self.nom_dep_map:
                dep = self.nom_dep_map[(pred, slot)]
        return dep

    def get_predicate_slots(self, event):
        """Get the possible slots for a predicate. In Propbank format,
        the slot would be arg0 to arg4 (and there are mapping to specific
        dependencies). In FrameNet format, the slots are determined from the
        frames.

        Args:
          event:

        Returns:

        """
        if self.frame_formalism == "FrameNet":
            frame = event["frame"]
            if not frame == -1 and frame in self.frame_slots:
                return self.frame_slots[frame]
        elif self.frame_formalism == "Propbank":
            pred = event["predicate"]
            if pred in self.frame_slots:
                return self.frame_slots[pred]
            else:
                # If we do not have a specific mapping, return arg0 to arg2.
                return [0, 1, 2]

        # Return an empty list set since we cannot handle this.
        return []
