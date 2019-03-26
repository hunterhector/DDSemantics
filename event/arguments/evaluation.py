import json
import os
from itertools import groupby
from operator import itemgetter
import logging


class ImplicitEval:
    def __init__(self, out_dir=None):
        self.num_instances = 0
        self.results = []
        self.out_dir = out_dir
        self.cutoffs = [1, 5, 10]

    def add_result(self, doc_id, event_idx, slot, score_labels, debug_data):
        self.results.append(
            (doc_id, event_idx, slot, score_labels, debug_data,)
        )

    def run(self):
        overall_res = {
            'num_docs': 0,
            'num_instances': 0,
            'scores': {},
            'gold': {},
        }

        for c in self.cutoffs:
            overall_res['scores']['p@%d' % c] = 0.0
            overall_res['scores']['r@%d' % c] = 0.0

            overall_res['gold']['p@%d' % c] = 0.0
            overall_res['gold']['r@%d' % c] = 0.0

        for doc_id, ref in groupby(self.results, itemgetter(0)):
            data = {
                'doc_id': doc_id,
                'predictions': [],
            }

            overall_res['num_docs'] += 1

            for _, event_idx, slot, score_labels, debug_data in ref:
                gold_rank = []
                top_k = []

                for r, (score, label) in enumerate(
                        sorted(score_labels, reverse=True)):
                    if label == 1:
                        gold_rank.append(r + 1)
                    if r < 5:
                        top_k.append(
                            (score, label, debug_data['entity_text'][r], r)
                        )

                num_correct = len(gold_rank)

                if num_correct == 0:
                    # No correct answer possible for this instance.
                    continue

                instance_res = {
                    'event_index': event_idx,
                    'slot': slot,
                    'num_gold': num_correct,
                    'gold_ranks': gold_rank,
                    'top_k': top_k,
                    'gold_entity': debug_data['gold_entity'],
                    'scores': {},
                    'gold': {},
                }

                overall_res['num_instances'] += 1

                for c in self.cutoffs:
                    tp = sum([1 if v <= c else 0 for v in gold_rank])
                    p = 1.0 * tp / c
                    r = 1.0 * tp / num_correct
                    instance_res['scores']['p@%d' % c] = p
                    instance_res['scores']['r@%d' % c] = r
                    instance_res['scores'][tp] = tp

                    gold_tp = min(num_correct, c)
                    gold_p = 1.0 * gold_tp / c
                    gold_r = gold_tp / num_correct

                    instance_res['gold']['p@%d' % c] = gold_p
                    instance_res['gold']['r@%d' % c] = gold_r

                    overall_res['scores']['p@%d' % c] += p
                    overall_res['scores']['r@%d' % c] += r

                    overall_res['gold']['p@%d' % c] += gold_p
                    overall_res['gold']['r@%d' % c] += gold_r

                data['predictions'].append(instance_res)

            if self.out_dir:
                detailed_path = os.path.join(self.out_dir, 'detailed_out.json')
                mode = 'a' if os.path.exists(detailed_path) else 'w'
                with open(detailed_path, mode) as res_out:
                    json.dump(data, res_out, indent=2)
                    res_out.write('\n')

        for k, v in overall_res['scores'].items():
            overall_res['scores'][k] = v / overall_res['num_instances']

        for k, v in overall_res['gold'].items():
            overall_res['gold'][k] = v / overall_res['num_instances']

        if self.out_dir:
            overall_path = os.path.join(self.out_dir, 'overall.json')
            with open(overall_path, 'w') as overall_out:
                json.dump(overall_res, overall_out, indent=2)
                overall_out.write('\n')
        else:
            logging.info("Test p@1 is %.4f." % overall_res['scores']['p@1'])

        return overall_res
