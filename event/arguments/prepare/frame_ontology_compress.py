from nltk.corpus import framenet as fn
import operator
import sys
import logging
from collections import Counter


class FrameCompressor:
    def __init__(self):
        self.h_ferel = {}
        self.__load_frame_relation()
        self.fe_mappings = {}
        self.fe_counts = Counter()

    def __load_frame_relation(self):
        for rel in fn.fe_relations():
            subFe = (rel.subFrame.name, rel.subFEName)
            superFe = (rel.superFrame.name, rel.superFEName)
            if rel.type.name in ['Subframe', 'Inheritance']:
                self.h_ferel[subFe] = superFe

    def load_fe_mapping(self, fe_mapping_file):
        not_found_fes = {}

        with open(fe_mapping_file) as fe_mapping:
            for line in fe_mapping:
                parts = line.strip().split('\t')

                if len(parts) < 2:
                    continue

                fe_info, arg_info = parts

                fe_parts = fe_info.split()
                frame_name = fe_parts[0]
                fe_count = int(fe_parts[-1])
                fe_name = "_".join(fe_parts[1:-1])

                fe = (frame_name, fe_name)

                seen_predicates = set()
                self.fe_counts[fe] += fe_count

                arg_fields = arg_info.split()
                args = []
                for arg in [arg_fields[x:x + 3] for x in
                            range(0, len(arg_fields), 3)]:
                    predicate, role, _ = arg
                    if not role == 'NA':
                        args.append(arg)

                    seen_predicates.add(predicate)

                args.sort(key=operator.itemgetter(2), reverse=True)

                if len(args) == 0:
                    not_found_fes[fe] = seen_predicates
                else:
                    self.fe_mappings[fe] = args

        return not_found_fes

    def use_parent_syntax(self, fe, target_pred):
        # fe_name = (frame, fe)
        if fe in self.h_ferel:
            super_fe = self.h_ferel[fe]
            if super_fe in self.fe_mappings:
                arg_roles = Counter()
                for predicate, arg, count in self.fe_mappings[super_fe]:
                    arg_roles[arg] += 1
                    if target_pred == predicate:
                        return arg

                sorted_roles = sorted(arg_roles.items(), reverse=True,
                                      key=operator.itemgetter(1))
                most_role = sorted_roles[0][0]
                return most_role
            else:
                return self.use_parent_syntax(super_fe, target_pred)
        return None

    def fill_blank(self, not_found_frames):
        print("Trying to map: %d" % len(not_found_frames))

        not_mapped = []

        for (frame_name, fe_name, _), predicates in not_found_frames.items():
            args = []
            found_map = False
            for predicate in predicates:
                arg = self.use_parent_syntax(
                    (frame_name, fe_name), predicate
                )
                if arg:
                    args.append((predicate, arg, 0))
                    found_map = True

            if not found_map:
                not_mapped.append((frame_name, fe_name))

        not_mapped_frames = [t[0] for t in not_mapped]

        print("Number not mappable: %d fe in %d frames." % (
            len(not_mapped), len(not_mapped_frames)))

        for fe in not_mapped:
            print("%s: %d\n" % (str(fe), self.fe_counts[fe]))

        print(not_mapped)

    def demo(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    frame_mapping_file = sys.argv[1]
    compressor = FrameCompressor()
    not_found_fes = compressor.load_fe_mapping(frame_mapping_file)
    logging.info("Loaded %d frame elements, %d frame elements not mapped.",
                 len(compressor.fe_mappings),
                 len(not_found_fes))
    compressor.fill_blank(not_found_fes)
