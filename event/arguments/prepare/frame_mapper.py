from nltk.corpus import framenet as fn
import operator
import sys
import logging
from collections import Counter, defaultdict


class FrameMapper:
    def __init__(self):
        self.h_ferel = {}
        self.__load_frame_relation()

        self.fe_mappings = {}
        self.arg_mappings = {}

        self.fe_counts = Counter()
        self.arg_counts = Counter()

    def __load_frame_relation(self):
        for rel in fn.fe_relations():
            subFe = (rel.subFrame.name, rel.subFEName)
            superFe = (rel.superFrame.name, rel.superFEName)
            if rel.type.name in ['Subframe', 'Inheritance']:
                self.h_ferel[subFe] = superFe

    def _write_mapping(self, out_file, mappings, counts):
        with open(out_file, 'w') as out:
            for fe, args in mappings.items():
                args_str = ["%s,%s:%d" % (arg[0], arg[1], arg[2]) for arg in
                            args]
                out.write('%s %s %d\t%s\n' % (
                    fe[0], fe[1], counts[fe], ' '.join(args_str)))

    def write_frame_mapping(self, frame_mapping_out, arg_mapping_out):
        self._write_mapping(frame_mapping_out, self.fe_mappings, self.fe_counts)
        self._write_mapping(arg_mapping_out, self.arg_mappings, self.arg_counts)

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

    def _load_mapping(self, path):
        not_found = {}
        mappings = {}
        counts = Counter()
        with open(path) as map_file:
            for line in map_file:
                parts = line.strip().split('\t')

                if len(parts) < 2:
                    continue

                fe_info, arg_info = parts
                frame_name, fe_name, fe_count = fe_info.split()
                fe = (frame_name, fe_name)

                seen_predicates = set()
                counts[fe] += int(fe_count)

                args = Counter()
                for arg_count in arg_info.split(' '):
                    arg, count = arg_count.split(":")
                    raw_predicate, role = arg.split(',')
                    pred_info = raw_predicate.split("_")
                    pred = pred_info[0]
                    if pred == 'not' and len(pred_info) > 1:
                        pred = pred_info[1]

                    if not role == 'NA':
                        args[(pred, role)] += int(count)
                    seen_predicates.add(pred)

                sorted_args = [(pred, role, count) for ((pred, role), count) in
                               sorted(args.items(), key=operator.itemgetter(1),
                                      reverse=True)]

                if len(sorted_args) == 0:
                    not_found[fe] = seen_predicates

                mappings[fe] = sorted_args

        return not_found, mappings, counts

    def load_raw_mapping(self, fe_mapping_file, arg_mapping_file):
        not_found_fes, self.fe_mappings, self.fe_counts = self._load_mapping(
            fe_mapping_file)
        not_found_args, self.arg_mappings, self.arg_counts = self._load_mapping(
            arg_mapping_file)

        filled_maps = self.fill_blank(not_found_fes)

        for (frame_name, fe_name), args in filled_maps.items():
            self.fe_mappings[frame_name, fe_name] = args

            print("Filling frame with ", frame_name, fe_name, args)

            for predicate, arg, _ in args:
                arg_tup = predicate, arg
                if arg_tup in self.arg_mappings:
                    self.arg_mappings[(predicate, arg)].append(
                        (frame_name, fe_name, 0)
                    )
                else:
                    self.arg_mappings[arg_tup] = [
                        (frame_name, fe_name, 0)
                    ]

    def use_parent_syntax(self, fe, target_pred):
        if fe in self.h_ferel:
            super_fe = self.h_ferel[fe]
            if super_fe in self.fe_mappings:
                arg_roles = Counter()
                for predicate, arg, count in self.fe_mappings[super_fe]:
                    arg_roles[arg] += 1
                    if target_pred == predicate:
                        return arg

                if len(arg_roles):
                    most_role = arg_roles.most_common()[0][0]
                    return most_role
                else:
                    return None
            else:
                return self.use_parent_syntax(super_fe, target_pred)
        return None

    def fill_blank(self, not_found_fes):
        print("%d frame elements not directly mapped." % len(not_found_fes))

        not_mapped = []

        filled_maps = {}

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
            else:
                filled_maps[(frame_name, fe_name)] = args

        not_mapped_frames = set([t[0] for t in not_mapped])

        print("After parent, %d FE in %d frames not mapped" % (
            len(not_mapped), len(not_mapped_frames)))

        return filled_maps


def main():
    import os
    work_dir = sys.argv[1]

    frame_mapping_file = os.path.join(work_dir, 'frames_args.tsv')
    arg_mapping_file = os.path.join(work_dir, 'args_frames.tsv')

    frame_mapping_out = os.path.join(work_dir, 'frames_args_filled.tsv')
    arg_mapping_out = os.path.join(work_dir, 'args_frames_filled.tsv')

    compressor = FrameMapper()
    compressor.load_raw_mapping(frame_mapping_file, arg_mapping_file)
    compressor.write_frame_mapping(frame_mapping_out, arg_mapping_out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
