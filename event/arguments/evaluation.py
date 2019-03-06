import json
import os
from itertools import groupby
from operator import itemgetter


class ImplicitEval:
    def __init__(self, out_dir):
        self.num_instances = 0
        self.results = []
        self.out_dir = out_dir
        self.cutoffs = [1, 5, 10]

    def add_result(self, doc_id, event_idx, slot_idx, gold_scores):
        gold_rank = []
        top_k = []
        for rank, (score, label) in enumerate(
                sorted(gold_scores, reverse=True)):
            if label == 1:
                gold_rank.append(rank + 1)
            if rank < 5:
                top_k.append((score, label))

        self.results.append((doc_id, event_idx, slot_idx, gold_rank, top_k))

    def run(self):
        detailed_path = os.path.join(self.out_dir, 'detailed_out.json')
        overall_path = os.path.join(self.out_dir, 'overall.json')

        with open(detailed_path, 'w') as res_out, \
                open(overall_path, 'w') as overall_out:

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

                for _, event_idx, slot_idx, l_gold_rank, l_top_k in ref:
                    num_correct = len(l_gold_rank)

                    if num_correct == 0:
                        # No correct answer possible for this instance.
                        continue

                    instance_res = {
                        'event_index': event_idx,
                        'slot_index': slot_idx,
                        'num_gold': num_correct,
                        'gold_ranks': l_gold_rank,
                        'top_k': l_top_k,
                        'scores': {},
                        'gold': {},
                    }

                    overall_res['num_instances'] += 1

                    for c in self.cutoffs:
                        tp = sum([1 if v <= c else 0 for v in l_gold_rank])
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

                json.dump(data, res_out, indent=2)
                res_out.write('\n')

            for k, v in overall_res['scores'].items():
                overall_res['scores'][k] = v / overall_res['num_instances']

            for k, v in overall_res['gold'].items():
                overall_res['gold'][k] = v / overall_res['num_instances']

            json.dump(overall_res, overall_out, indent=2)
            overall_out.write('\n')
