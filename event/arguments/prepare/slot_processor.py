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
    if dep == 'nsubj' or dep == 'agent' or dep == 'subj':
        return 'subj'
    elif dep == 'dobj' or dep == 'nsubjpass' or dep == 'obj':
        return 'obj'
    elif dep == 'iobj':
        # iobj is more prep like location
        return 'prep'
    elif dep.startswith('prep'):
        return 'prep'
    return 'NA'


def sort_arg_priority(args, pred_start, pred_end):
    # TODO priority didn't consider sources from GoldStandard
    p_arg_list = []

    for i, (dep, fe, a, source) in enumerate(args):
        arg_start, arg_end = a['arg_start'], a['arg_end']

        if arg_start >= pred_end:
            dist = arg_start - pred_end
        elif arg_end <= pred_start:
            dist = pred_start - arg_end
        else:
            # Overlapping case, we give the lowest priority.
            dist = float('inf')

        num_emtpy = 0
        if dep == 'None':
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
        p_arg_list.append((priority, (dep, fe, a)))

    return sorted(p_arg_list, key=itemgetter(0))


class SlotHandler:
    def __init__(self, frame_dir, frame_dep_map, dep_frame_map,
                 nombank_arg_map):
        self.frame_priority = self.__load_fe_orders(frame_dir)
        self.frame_deps, self.frame_counts = self.__load_frame_map(
            frame_dep_map)

        self.nombank_mapping = self.__load_nombank_map(
            nombank_arg_map
        )

        self.dep_frames, self.dep_counts = self.__load_frame_map(dep_frame_map)

    @staticmethod
    def __load_nombank_map(nombank_arg_map):
        nom_map = {}

        with open(nombank_arg_map) as map_file:
            for line in map_file:
                fields = line.strip().split('\t')
                if line.startswith("#"):
                    role_names = fields[2:]
                else:
                    nom_form = fields[0]
                    verb_form = fields[1]

                    nom_map[nom_form] = (verb_form, {})

                    for role_name, dep_name in zip(role_names, fields[2:]):
                        if not dep_name == '-':
                            nom_map[nom_form][1][role_name] = dep_name
        return nom_map

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
                arg_list.append((dep, full_fe, arg, 'origin'))
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
                arg_list.append((dep, (frame, fe), arg, 'origin'))
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
            else:
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
            nombank_special_dep = None
            if predicate in self.nombank_mapping and 'propbank_role' in arg:
                this_nom_map = self.nombank_mapping[predicate][1]
                prop_role = arg['propbank_role'].replace('i_', '')

                if prop_role in this_nom_map:
                    nombank_special_dep = this_nom_map[prop_role]

            if nombank_special_dep is None:
                position = get_dep_position(dep)
            else:
                position = get_dep_position(nombank_special_dep)

            arg_candidates[position].append((dep, full_fe, arg, source))

        # TODO: still some hairy stuff like duplicate FE slots.

        # Final step, organize all the args.
        final_args = {}
        unsure_args = []
        for position, candidate_args in arg_candidates.items():
            p_arg_info = sort_arg_priority(
                candidate_args,
                event['predicate_start'],
                event['predicate_end']
            )

            if predicate == 'leak':
                from pprint import pprint
                pprint(arg_candidates)

                pprint(p_arg_info)
                input('check the sorted list')

            if position == 'NA':
                unsure_args = [p[1] for p in p_arg_info]
            else:
                final_args[position] = [p[1] for p in p_arg_info]

        # Put all the unsured ones to the last bin.
        final_args['prep'].extend(unsure_args)

        return final_args
