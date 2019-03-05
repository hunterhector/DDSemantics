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
        for rank, (score, label) in gold_scores:
            if label == 1:
                gold_rank.append(rank + 1)
            if rank < 5:
                top_k.append((score, label))

        self.results.append((doc_id, event_idx, slot_idx, gold_rank, top_k))

    def run(self):
        with open(os.path.join(self.out_dir, 'test_output.json')) as res_out:
            for doc_id, ref in groupby(self.results, itemgetter(0)):
                data = {
                    'doc_id': doc_id,
                    'predictions': [],
                }

                for event_idx, slot_idx, l_gold_rank, l_top_k in ref:
                    instance_out = {
                        'event_index': event_idx,
                        'slot_index': slot_idx,
                        'gold_ranks': l_gold_rank,
                        'top_k': l_top_k,
                        'scores': {},
                    }

                    num_correct = len(l_gold_rank)

                    for wh in self.cutoffs:
                        tp = sum([1 if v <= wh else 0 for v in l_gold_rank])
                        p = 1.0 * tp / wh
                        r = 1.0 * tp / num_correct
                        instance_out['scores']['p@%d' % wh] = p
                        instance_out['scores']['r@%d' % wh] = r
                    data['predictions'].append(instance_out)

                json.dump(data, res_out, indent=2)
                res_out.write('\n')

        with open(os.path.join(self.out_dir, 'overall.json')) as out:

            out.write('\n')
