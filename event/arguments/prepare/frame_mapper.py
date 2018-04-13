from nltk.corpus import framenet as fn
import operator
import sys
import logging
from collections import Counter


class FrameMapper:
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

    def write_frame_mapping(self, frame_mapping_out):
        with open(frame_mapping_out, 'w') as out:
            for fe, args in self.fe_mappings.items():
                args_str = ["%s %s %d" % (arg[0], arg[1], arg[2]) for arg in
                            args]
                out.write('%s %s %d\t%s\n' % (
                    fe[0], fe[1], self.fe_counts[fe], ' '.join(args_str)))

    def get_mapping(self, mapping_file):
        frame_args = {}
        args_frames = {}
        with open(mapping_file) as mapping:
            for line in mapping:
                fe_info, arg_info = line.split('\t')
                arg_fields = arg_info.split()

                frame_name, fe_name, fe_count = fe_info.split()
                fe = (frame_name, fe_name)

                args = []
                for arg in [arg_fields[x:x + 3] for x in
                            range(0, len(arg_fields), 3)]:
                    raw_predicate, role, arg_count = arg
                    args.append(arg)

                frame_args[fe] = args
        return frame_args, args_frames

    def load_raw_mapping(self, fe_mapping_file):
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

                if frame_name == "NA" or fe_name == "NA":
                    continue

                fe = (frame_name, fe_name)

                seen_predicates = set()
                self.fe_counts[fe] += fe_count

                arg_fields = arg_info.split()
                args = Counter()
                for arg in [arg_fields[x:x + 3] for x in
                            range(0, len(arg_fields), 3)]:
                    raw_predicate, role, arg_count = arg

                    pred_info = raw_predicate.split("_")
                    pred = pred_info[0]
                    if pred == 'not' and len(pred_info) > 1:
                        pred = pred_info[1]

                    if not role == 'NA':
                        args[(pred, role)] += int(arg_count)

                    seen_predicates.add(pred)

                sorted_args = sorted(args.items(), key=operator.itemgetter(1),
                                     reverse=True)
                sorted_args = [(pred, role, count) for ((pred, role), count) in
                               sorted_args]

                if len(sorted_args) == 0:
                    not_found_fes[fe] = seen_predicates
                else:
                    self.fe_mappings[fe] = sorted_args

        self.fill_blank(not_found_fes)

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

    def fill_blank(self, not_found_fes):
        print("%d frame elements not directly mapped." % len(not_found_fes))

        not_mapped = []

        for (frame_name, fe_name), predicates in not_found_fes.items():
            args = []
            found_mapping = False
            for predicate in predicates:
                arg = self.use_parent_syntax(
                    (frame_name, fe_name), predicate
                )
                if arg:
                    args.append((predicate, arg, 0))
                    found_mapping = True

            if not found_mapping:
                not_mapped.append((frame_name, fe_name))

        not_mapped_frames = set([t[0] for t in not_mapped])

        print("After parent, %d FE in %d frames not mapped" % (
            len(not_mapped), len(not_mapped_frames)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    frame_mapping_file = sys.argv[1]
    frame_mapping_out = sys.argv[2]
    arg_mapping_out = sys.argv[3]

    compressor = FrameMapper()
    compressor.load_raw_mapping(frame_mapping_file)
    compressor.write_frame_mapping(frame_mapping_out)
