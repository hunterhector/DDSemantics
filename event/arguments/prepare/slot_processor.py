import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from event.arguments import util
from operator import itemgetter


def remove_slot_info(arg_info):
    content = {}
    for k, v in arg_info.items():
        if not (k == 'dep' or k == 'fe'):
            content[k] = v
    return content


def get_dep_position(dep):
    if dep == 'nsubj' or dep == 'agent':
        return 'subj'
    elif dep == 'dobj' or dep == 'nsubjpass':
        return 'obj'
    elif dep == 'iobj':
        # iobj is more prep like location
        return 'prep'
    elif dep.startswith('prep_'):
        return 'prep'
    return 'NA'


def sort_arg_priority(args, pred_start, pred_end):
    p_arg_list = []

    for i, (d, fe, a, source) in enumerate(args):
        arg_start, arg_end = a['arg_start'], a['arg_end']

        if arg_start >= pred_end:
            dist = arg_start - pred_end
        elif arg_end <= pred_start:
            dist = pred_start - arg_end
        else:
            # Overlapping case, we give the lowest priority.
            dist = float('inf')

        num_emtpy = 0
        if d == 'None':
            num_emtpy += 1

        if fe == 'None':
            num_emtpy += 1

        if source == 'origin':
            source_level = 0
        elif source.startswith('imputed_'):
            source_level = 1
        else:
            source_level = 2

        # Large number correspond to lower priority.
        # Corresponds to: source_level, empty_level, distance
        priority = (source_level, num_emtpy, dist)
        p_arg_list.append((priority, (d, fe, a)))

    return sorted(p_arg_list)


class SlotHandler:
    def __init__(self, frame_dir, frame_dep_map, dep_frame_map):
        self.frame_priority = self.__load_fe_orders(frame_dir)
        self.frame_deps, self.frame_counts = self.__load_frame_map(
            frame_dep_map)
        self.dep_frames, self.dep_counts = self.__load_frame_map(dep_frame_map)

    @staticmethod
    def __load_fe_orders(frame_dir):
        frame_priority = {}

        for f in os.listdir(frame_dir):
            if f.endswith(".xml"):
                with open(os.path.join(frame_dir, f)) as frame_file:
                    frame = ET.parse(frame_file).getroot()
                    frame_name = frame.attrib['name']
                    frame_priority[frame_name] = []

                    for fe in frame.findall(
                            '{http://framenet.icsi.berkeley.edu}FE'):
                        frame_priority[frame_name].append(
                            {
                                'fe_name': fe.attrib['name'],
                                'core_type': fe.attrib['coreType'],
                            }
                        )
        return frame_priority

    @staticmethod
    def __load_frame_map(frame_map_file):
        fmap = {}
        counts = {}
        with open(frame_map_file) as frame_maps:
            for line in frame_maps:
                from_info, target_info = line.split('\t')
                from_pred, from_arg, from_count = from_info.split(' ')
                fmap[from_pred, from_arg] = []
                for target in target_info.split():
                    role_info, count = target.split(':')
                    role_parts = role_info.split(',')

                    to_pred = ','.join(role_parts[:-1])
                    to_arg = role_parts[-1]

                    fmap[from_pred, from_arg].append(
                        (to_pred, to_arg, int(count)))
                counts[from_pred] = from_count
        return fmap, counts

    def impute_fe(self, arg_list, predicate, dep_slots, frame_slots):
        imputed_fes = defaultdict(Counter)

        for dep, (full_fe, arg) in dep_slots.items():
            imputed = False
            candidates = self.dep_frames.get((predicate, dep), [])
            for cand_frame, cand_fe, cand_count in candidates:
                if (cand_frame, cand_fe) not in frame_slots:
                    imputed_fes[(cand_frame, cand_fe)][dep] = cand_count
                    imputed = True
                    break
            if not imputed:
                arg_list.append(dep, full_fe, arg, 'origin')
        return imputed_fes

    def impute_deps(self, arg_list, predicate, dep_slots, frame_slots):
        imputed_deps = defaultdict(Counter)

        for (frame, fe), (dep, arg) in frame_slots.items():
            imputed = False
            for pred, dep, cand_count in self.frame_deps.get((frame, fe), []):
                if dep not in dep_slots and pred == predicate:
                    imputed_deps[dep][(frame, fe)] = cand_count
                    imputed = True
                    break
            if not imputed:
                arg_list.append(dep, (frame, fe), arg, 'origin')
        return imputed_deps

    def organize_args(self, event):
        # Step 0, read slot information
        args = event['arguments']

        arg_list = []

        dep_slots = {}
        frame_slots = {}

        predicate = util.remove_neg(event.get('predicate'))
        frame = event.get('frame', 'NA')
        for arg in args:
            dep = arg.get('dep', 'NA')
            fe = arg.get('fe', 'NA')

            plain_arg = remove_slot_info(arg)

            if fe == 'NA' and not dep == 'NA':
                if not dep.startswith('prep_'):
                    dep_slots[dep] = ((frame, fe), remove_slot_info(arg))
            elif dep == 'NA' and not fe == 'NA':
                frame_slots[(frame, fe)] = (dep, remove_slot_info(arg))
            elif not dep == 'NA' and not fe == 'NA':
                arg_list.append((dep, (frame, fe), plain_arg, 'origin'))

        # Step 1, impute dependency or FE slot.
        imputed_fes = self.impute_fe(arg_list, predicate, dep_slots,
                                     frame_slots)
        imputed_deps = self.impute_deps(arg_list, predicate, dep_slots,
                                        frame_slots)

        # Step 2, find consistent imputation with priority and counts.
        for full_fe, dep_counts in imputed_fes.items():
            dep, count = dep_counts.most_common(1)[0]
            _, arg = dep_slots[dep]
            arg_list.append((dep, full_fe, arg, 'imputed_fes'))

        for i_dep, frame_counts in imputed_deps.items():
            full_fe, count = frame_counts.most_common(1)[0]
            _, arg = frame_slots[full_fe]
            arg_list.append((i_dep, full_fe, arg, 'imputed_deps'))

        arg_candidates = {
            'subj': [],
            'obj': [],
            'prep': [],
            'NA': [],
        }

        for dep, full_fe, arg, source in arg_list:
            arg_candidates[get_dep_position(dep)].append(
                (dep, full_fe, arg, source)
            )

        # Final step, organize all the args.
        final_args = {}
        unsure_args = []
        for position, candidate_args in arg_candidates.items():
            p_arg_info = sort_arg_priority(
                candidate_args,
                event['predicate_start'],
                event['predicate_end']
            )
            if position == 'NA':
                unsure_args = [p[1] for p in p_arg_info]
            else:
                final_args[position] = [p[1] for p in p_arg_info]

        # Put all the unsured ones to the last bin.
        final_args['prep'].extend(unsure_args)

        # import pprint
        # pprint.pprint('initial ')
        # pprint.pprint(args)
        # pprint.pprint('final args')
        # pprint.pprint(final_args)
        #
        # input('wait')

        return final_args


def main():
    handler = SlotHandler('/home/zhengzhl/resources/fndata-1.5/frame/')

    while True:
        f = input('Input frame:')
        p = handler.frame_priority.get(f, None)
        if not p:
            continue
        else:
            print(p)


if __name__ == '__main__':
    main()
