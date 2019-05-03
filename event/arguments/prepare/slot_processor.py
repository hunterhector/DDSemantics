import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from event.arguments import util


class SlotHandler:
    def __init__(self, frame_dir):
        self.frame_priority = self.__load_fe_orders(frame_dir)

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

    # def tiebreak_arg(self, tied_args, pred_start, pred_end):
    #     top_index = 0
    #     priority = (3, float('inf'))
    #
    #     # TODO: priority didn't consider the source.
    #     for i, (d, ffe, a, source) in enumerate(tied_args):
    #         arg_start, arg_end = a['arg_start'], a['arg_end']
    #
    #         if arg_start >= pred_end:
    #             dist = arg_start - pred_end
    #         elif arg_end <= pred_start:
    #             dist = pred_end - arg_end
    #         else:
    #             # Overlapping case, we give the lowest priority.
    #             dist = float('inf')
    #
    #         num_emtpy = 0
    #         if d == 'None':
    #             num_emtpy += 1
    #
    #         if ffe == 'None':
    #             num_emtpy += 1
    #
    #         this_priority = (num_emtpy, dist)
    #
    #         if this_priority < priority:
    #             top_index = i
    #             priority = this_priority
    #
    #     return tied_args[top_index]
    #
    # def impute_args(self, event, frame_args, arg_frames):
    #     args = event['arguments']
    #     pred_start, pred_end = event['predicate_start'], event['predicate_end']
    #
    #     arg_candidates = {
    #         'subj': [],
    #         'obj': [],
    #         'prep': [],
    #     }
    #
    #     dep_slots = {}
    #     frame_slots = {}
    #
    #     predicate = util.remove_neg(event.get('predicate'))
    #     frame = event.get('frame', 'NA')
    #
    #     for arg in args:
    #         dep = arg.get('dep', 'NA')
    #         fe = arg.get('fe', 'NA')
    #
    #         if not dep == 'NA' and self.get_dep_position(dep) not in arg_candidates:
    #             # If dep is an known but not in our target list, ignore them.
    #             continue
    #
    #         if not dep == 'NA':
    #             dep_slots[dep] = ((frame, fe), remove_slot_info(arg))
    #
    #         if not fe == 'NA':
    #             frame_slots[(frame, fe)] = (dep, remove_slot_info(arg))
    #
    #     imputed_fes = defaultdict(Counter)
    #     for dep, (full_fe, arg) in dep_slots.items():
    #         position = get_dep_position(dep)
    #
    #         if full_fe[1] == 'NA':
    #             candidates = arg_frames.get((predicate, dep), [])
    #             not_trust = dep.startswith('prep_')
    #             imputed = False
    #
    #             if not not_trust:
    #                 for cand_frame, cand_fe, cand_count in candidates:
    #                     if (cand_frame, cand_fe) not in frame_slots:
    #                         imputed_fes[(cand_frame, cand_fe)][dep] = cand_count
    #                         imputed = True
    #                         break
    #
    #             if not imputed:
    #                 # No impute can be found, or we do not trust the imputation.
    #                 # In this case, we place an empty FE name here.
    #                 arg_candidates[position].append(
    #                     (dep, None, arg, 'no_impute'))
    #         else:
    #             arg_candidates[position].append((dep, full_fe, arg, 'origin'))
    #
    #     imputed_deps = defaultdict(Counter)
    #     for (frame, fe), (dep, arg) in frame_slots.items():
    #         if dep == 'NA':
    #             for pred, dep, cand_count in frame_args.get((frame, fe), []):
    #                 if dep not in dep_slots and pred == event['predicate']:
    #                     imputed_deps[dep][(frame, fe)] = cand_count
    #                     break
    #
    #     for full_fe, dep_counts in imputed_fes.items():
    #         dep, count = dep_counts.most_common(1)[0]
    #         _, arg = dep_slots[dep]
    #         position = get_dep_position(dep)
    #         arg_candidates[position].append((dep, full_fe, arg, 'deps'))
    #
    #     for i_dep, frame_counts in imputed_deps.items():
    #         full_fe, count = frame_counts.most_common(1)[0]
    #         position = get_dep_position(i_dep)
    #         _, arg = frame_slots[full_fe]
    #         if position == 'NA':
    #             if 'prep' not in arg_candidates:
    #                 arg_candidates['prep'].append(
    #                     (i_dep, full_fe, arg, 'frames'))
    #
    #     final_args = {}
    #     for position, candidate_args in arg_candidates.items():
    #         if len(candidate_args) > 1:
    #             a = tiebreak_arg(candidate_args, pred_start, pred_end)
    #             # Here we only take the first 3.
    #             final_args[position] = a[:3]
    #         elif len(candidate_args) == 1:
    #             final_args[position] = candidate_args[0][:3]
    #         else:
    #             final_args[position] = None
    #
    #     return final_args


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
