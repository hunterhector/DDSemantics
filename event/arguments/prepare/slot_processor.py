import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from event.arguments import util
from operator import itemgetter
from pprint import pprint


def remove_slot_info(arg_info):
    content = {}
    for k, v in arg_info.items():
        if not (k == 'dep' or k == 'fe'):
            content[k] = v
    return content


def get_position(dep):
    simple_dep = get_simple_dep(dep)

    if simple_dep == 'dep':
        # Put other dependency to the prepositional slot in the fixed
        # mode.
        return 'prep'
    elif simple_dep.startswith('prep_'):
        return 'prep'
    else:
        return simple_dep


def get_simple_dep(dep):
    if dep == 'nsubj' or dep == 'agent' or dep == 'subj':
        return 'subj'
    elif dep == 'dobj' or dep == 'nsubjpass' or dep == 'obj':
        return 'obj'
    elif dep == 'iobj':
        # iobj is more prep like location
        return 'prep'
    elif dep.startswith('prep_'):
        return dep
    elif dep.startswith('prepc_'):
        return dep.replace('prepc_', 'prep_')
    elif dep == 'NA':
        return 'NA'
    else:
        # A generic "other" dependency
        return 'dep'


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

    return [p[1] for p in sorted(p_arg_list, key=itemgetter(0))]


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

    def get_most_freq_dep(self, predicate, frame, fe):
        if (frame, fe) in self.frame_deps:
            for pred, dep, count in self.frame_deps[(frame, fe)]:
                if pred == predicate:
                    return dep
        return None

    def impute_fe(self, arg_list, predicate, frame, dep_slots, fe_slots):
        imputed_fes = defaultdict(Counter)

        for dep, args in dep_slots.items():
            for arg in args:
                for cand_frame, cand_fe, cand_count in self.dep_frames.get(
                        (predicate, dep), []):
                    if cand_fe not in fe_slots and cand_frame == frame:
                        imputed_fes[cand_fe][dep] = cand_count
                        break
                else:
                    arg_list.append((dep, 'NA', arg, 'origin'))
        return imputed_fes

    def impute_deps(self, arg_list, predicate, frame, dep_slots, frame_slots):
        imputed_deps = defaultdict(Counter)

        for fe, args in frame_slots.items():
            for arg in args:
                for cand_pred, cand_dep, cand_count in self.frame_deps.get(
                        (frame, fe), []):
                    if cand_dep not in dep_slots and cand_pred == predicate:
                        imputed_deps[cand_dep][fe] = cand_count
                        break
                else:
                    arg_list.append(('NA', fe, arg, 'origin'))
        return imputed_deps

    def organize_args(self, event):
        # Step 0, read slot information
        args = event['arguments']
        # TODO: handle slashed events (small-investor)
        predicate = util.remove_neg(event.get('predicate'))
        frame = event.get('frame', 'NA')

        # List of resolved arguments.
        arg_list = []

        # Arguments that have a valid dependency.
        # dep_slots = {}
        dep_slots = defaultdict(list)
        # Arguments that have a valid frame element.
        # frame_slots = {}
        fe_slots = defaultdict(list)

        test_arg_list = []

        for arg in args:
            dep = arg['dep']
            fe = arg['fe']

            # There are a few defined mapping for the target predicates.
            if predicate in self.nombank_mapping and 'propbank_role' in arg:
                this_nom_map = self.nombank_mapping[predicate][1]
                prop_role = arg['propbank_role'].replace('i_', '')
                if prop_role in this_nom_map:
                    dep = this_nom_map[prop_role]

            plain_arg = remove_slot_info(arg)

            if fe == 'NA' and not dep == 'NA' and not dep.startswith('prep_'):
                # Try to impute the FE from dependency label.
                # TODO: The bug is here, when we map arguments here,
                #  if the dependency is the same then we lose a lot.
                dep_slots[dep].append(plain_arg)
            elif dep == 'NA' and not fe == 'NA':
                # Try to impute the dependency from FE
                fe_slots[fe].append(plain_arg)
            else:
                # Add the others directly to the list.
                arg_list.append((dep, fe, plain_arg, 'origin'))
                test_arg_list.append((dep, fe, plain_arg, 'origin'))

        # Step 1, impute dependency or FE slot.
        imputed_fes = self.impute_fe(
            arg_list, predicate, frame, dep_slots, fe_slots)
        imputed_deps = self.impute_deps(
            arg_list, predicate, frame, dep_slots, fe_slots)

        # Step 2, find consistent imputation with priority and counts.
        for fe, dep_counts in imputed_fes.items():
            most_common = True
            for dep, count in dep_counts.most_common():
                for arg in dep_slots[dep]:
                    if most_common:
                        arg_list.append((dep, fe, arg, 'imputed_fes'))
                    else:
                        arg_list.append((dep, 'NA', arg, 'imputed_fes'))
                most_common = False

        for i_dep, frame_counts in imputed_deps.items():
            most_common = True
            for fe, count in frame_counts.most_common():
                for arg in fe_slots[fe]:
                    if most_common:
                        arg_list.append((i_dep, fe, arg, 'origin'))
                    else:
                        arg_list.append(('NA', fe, arg, 'origin'))
                most_common = False

        arg_candidates = {
            'subj': [],
            'obj': [],
            'prep': [],
            'NA': [],
        }

        for dep, fe, arg, source in arg_list:
            position = get_position(dep)
            arg_candidates[position].append((dep, fe, arg, source))

        # Final step, organize all the args.
        final_args = {}
        unsure_args = []
        for position, candidate_args in arg_candidates.items():
            sorted_arg_info = sort_arg_priority(
                candidate_args,
                event['predicate_start'],
                event['predicate_end']
            )

            if position == 'NA':
                unsure_args = sorted_arg_info
            else:
                final_args[position] = sorted_arg_info

        # Put all the unsure ones to the last bin.
        final_args['prep'].extend(unsure_args)
        return final_args
