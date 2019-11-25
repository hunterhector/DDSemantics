from nltk.corpus import framenet as fn
import operator
import sys
import logging
from collections import Counter, defaultdict

import event.util
from event.arguments import util


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
            for fe, args_count in mappings.items():
                if len(args_count) > 0:
                    args_str = ["%s,%s:%d" % (arg[0], arg[1], count) for
                                arg, count in args_count.items()]
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

    def _load_mapping(self, path, dep_source=False, dep_target=False):
        not_found = defaultdict(set)
        mappings = defaultdict(Counter)
        counts = Counter()
        with open(path) as map_file:
            for line in map_file:
                parts = line.strip().split('\t')

                if len(parts) < 2:
                    continue

                from_role_info, to_role_info = parts

                info_parts = from_role_info.split()
                if not len(info_parts) == 3:
                    continue

                from_frame_name, from_arg_name, from_count = info_parts

                if dep_source:
                    from_frame_name = event.util.remove_neg(from_frame_name)

                from_arg = (from_frame_name, from_arg_name)

                seen_predicates = set()
                counts[from_arg] += int(from_count)

                args = Counter()
                for arg_count in to_role_info.split(' '):
                    parts = arg_count.split(":")
                    if len(parts) == 2:
                        arg, count = parts
                        arg_parts = arg.split(',')
                        predicate = ','.join(arg_parts[:-1])
                        role = arg_parts[-1]

                        if dep_target:
                            predicate = event.util.remove_neg(predicate)

                        if not role == 'NA':
                            args[(predicate, role)] += int(count)
                        seen_predicates.add(predicate)

                mappings[from_arg] += args

                if len(args) == 0:
                    not_found[from_arg] |= seen_predicates

        for key, seen_pred in not_found.items():
            if key in mappings:
                for (mapped_pred, mapped_arg), count in mappings[key].items():
                    if mapped_pred in seen_pred:
                        seen_pred.remove(mapped_pred)
                        # print('{} is mapped to {}:{} for {} times, but '
                        #       'treated unseen, now removed'
                        #       .format(key, mapped_pred, mapped_arg, count))

        return not_found, mappings, counts

    def load_raw_mapping(self, fe_mapping_file, arg_mapping_file):
        print("Loading from " + arg_mapping_file)
        not_found_args, self.arg_mappings, self.arg_counts = self._load_mapping(
            arg_mapping_file, dep_source=True)

        print("Loading from " + fe_mapping_file)
        not_found_fes, self.fe_mappings, self.fe_counts = self._load_mapping(
            fe_mapping_file, dep_target=True)

        filled_maps = self.fill_blank(not_found_fes)

        for (frame_name, fe_name), args in filled_maps.items():
            self.fe_mappings[frame_name, fe_name] = args

            for (predicate, arg) in args.keys():
                self.arg_mappings[(predicate, arg)][(frame_name, fe_name)] = 0

    def use_parent_syntax(self, fe, target_pred):
        if fe in self.h_ferel:
            super_fe = self.h_ferel[fe]
            if super_fe in self.fe_mappings:
                arg_roles = Counter()
                for (predicate, arg), count in self.fe_mappings[
                    super_fe].items():
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
            args = Counter()
            found_mapping = False
            for predicate in predicates:
                arg = self.use_parent_syntax(
                    (frame_name, fe_name), predicate
                )
                if arg:
                    args[predicate, arg] = 0
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
