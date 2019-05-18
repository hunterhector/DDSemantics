import json
import os
from itertools import groupby
from operator import itemgetter
import logging
from event import util

from event.arguments.cloze_readers import ghost_entity_text
from pprint import pprint
from collections import Counter, defaultdict


class ImplicitEval:
    def __init__(self, slot_names, out_dir=None):
        self.num_instances = 0
        self.results = []
        self.out_dir = out_dir
        self.cutoffs = [1, 5, 10]

        self.slot_names = slot_names

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            self.detail_path = os.path.join(self.out_dir, 'detailed_out.json')
            self.overall_path = os.path.join(self.out_dir, 'overall.json')
            if os.path.exists(self.detail_path):
                util.append_num_to_path(self.detail_path)

        self.overall_res = {
            'num_docs': 0,
            'num_instances': 0,
            'num_fillable': 0,
            'num_fill_attempts': Counter(),
            'tp': Counter(),
            'scores': {},
            'gold': {},
        }

        self.selectors = self.candidate_selectors()
        self.k = 5

        for name in self.selectors.keys():
            self.overall_res['scores'][name] = {}
            self.overall_res['gold'][name] = {}
            for c in self.cutoffs:
                self.overall_res['scores'][name][f'p@{c}'] = 0.0
                self.overall_res['scores'][name][f'r@{c}'] = 0.0

                self.overall_res['gold'][name][f'p@{c}'] = 0.0
                self.overall_res['gold'][name][f'r@{c}'] = 0.0

    def add_prediction(self, doc_id, event_indexes, slot_indexes, coh_scores,
                       gold_labels, candidate_meta, instance_meta):
        for (((event_idx, slot_idx), result), ins_meta) in zip(groupby(
                zip(zip(event_indexes, slot_indexes),
                    zip(coh_scores, gold_labels),
                    candidate_meta, ),
                key=itemgetter(0)), instance_meta):
            _, score_labels, c_meta = zip(*result)
            self.add_result(
                doc_id, event_idx, slot_idx, score_labels, ins_meta, c_meta
            )

    @staticmethod
    def candidate_selectors():
        selector = {}

        def neighbor_selector(meta):
            if 0 <= meta['distance_to_event'] <= 2:
                return True
            else:
                return False

        def gold_selector(meta):
            if meta['source'] == 'gold':
                return True
            else:
                return False

        def neighbor_gold_selector(meta):
            return neighbor_selector(meta) and gold_selector(meta)

        def all_selector(meta):
            return True

        selector['neighbor'] = neighbor_selector
        selector['gold'] = gold_selector
        selector['neighbor_gold'] = neighbor_gold_selector
        selector['all'] = all_selector

        return selector

    def add_result(self, doc_id, event_idx, slot_idx, score_labels, ins_meta,
                   c_meta):
        self.results.append(
            (doc_id, event_idx, slot_idx, score_labels, ins_meta, c_meta,)
        )

        data = {
            'doc_id': doc_id,
            'results': [],
            'predictions': [],
        }

        self.overall_res['num_docs'] += 1

        ranked_predictions = []

        sorted_result = sorted(zip(score_labels, c_meta), reverse=True,
                               key=itemgetter(0))

        num_golds = Counter()
        gold_ranks = defaultdict(list)
        top_k = defaultdict(list)

        rank_count = Counter()
        for (score, label), meta in sorted_result:
            ranked_predictions.append(
                {
                    'score': score,
                    'label': label,
                    'meta': meta,
                    'distance_to_event': meta['distance_to_event'],
                    'source': meta['source'],
                }
            )

            for sel_name, selector in self.candidate_selectors().items():
                rank_count[sel_name] += 1
                if selector(meta):
                    if label == 1:
                        gold_ranks[sel_name].append(rank_count[sel_name])
                        num_golds[sel_name] += 1

                    if rank_count[sel_name] < self.k:
                        top_k[sel_name].append(
                            {
                                'meta': meta,
                                'rank': rank_count[sel_name],
                                'score': score,
                            }
                        )

                    if sel_name == 'neighbor':
                        print(meta)

                    input('these are close by entities')

        if ins_meta['has_true']:
            self.overall_res['num_fillable'] += 1

        for sel_name, tops in top_k.items():
            if tops[0]['meta']['entity'] == ghost_entity_text:
                self.overall_res['num_fill_attempts'][sel_name] += 1

        instance_res = {
            'event_index': event_idx,
            'predicate': ins_meta['predicate'],
            'slot_index': slot_idx,
            'gold_entity': ins_meta['gold_entity'],
            'slot_name': self.slot_names[slot_idx],
            'num_gold': num_golds,
            'gold_ranks': dict(gold_ranks),
            'top_k': {},
            'scores': dict([(n, {}) for n in self.selectors.keys()]),
            'gold': dict([(n, {}) for n in self.selectors.keys()]),
        }

        for sel_name, ranks in gold_ranks.items():
            instance_res['top_k'][sel_name] = top_k[sel_name]
            # If one of the gold instance is ranked as the best.
            if any([r == 1 for r in gold_ranks[sel_name]]):
                self.overall_res['tp'][sel_name] += 1

        self.overall_res['num_instances'] += 1

        for c in self.cutoffs:
            for sel_name, num_gold in num_golds.items():
                if num_gold == 0:
                    p = 0
                    r = 0
                    gold_p = 0
                    gold_r = 0
                else:
                    tp_at_c = sum(
                        [1 if v <= c else 0 for v in gold_ranks[sel_name]]
                    )
                    p = 1.0 * tp_at_c / c
                    r = 1.0 * tp_at_c / num_gold
                    gold_tp = min(num_gold, c)
                    gold_p = 1.0 * gold_tp / c
                    gold_r = gold_tp / num_gold

                instance_res['scores'][sel_name]['p@%d' % c] = p
                instance_res['scores'][sel_name]['r@%d' % c] = r

                instance_res['gold'][sel_name]['p@%d' % c] = gold_p
                instance_res['gold'][sel_name]['r@%d' % c] = gold_r

                self.overall_res['scores'][sel_name][f'p@{c}'] += p
                self.overall_res['scores'][sel_name][f'r@{c}'] += r

                self.overall_res['gold'][sel_name][f'p@{c}'] += gold_p
                self.overall_res['gold'][sel_name][f'r@{c}'] += gold_r

        data['results'].append(instance_res)
        data['predictions'] = ranked_predictions

        if self.out_dir:
            mode = 'a' if os.path.exists(self.detail_path) else 'w'
            with open(self.detail_path, mode) as res_out:
                json.dump(data, res_out, indent=2)
                res_out.write('\n')

    def collect(self):
        for sel_name, sel_scores in self.overall_res['scores'].items():
            for k, v in sel_scores.items():
                if self.overall_res['num_instances'] > 0:
                    self.overall_res['scores'][sel_name][k] /= self.overall_res[
                        'num_instances']
                else:
                    self.overall_res['scores'][k] = 0

        for sel_name, sel_scores in self.overall_res['gold'].items():
            for k, v in sel_scores.items():
                if self.overall_res['num_instances'] > 0:
                    self.overall_res['gold'][sel_name][k] /= self.overall_res[
                        'num_instances']
                else:
                    self.overall_res['gold'][k] = 0

        precs = {}
        for sel_name, tp in self.overall_res['scores'].items():
            if self.overall_res['num_fillable'] > 0:
                prec = self.overall_res['tp'][sel_name] / self.overall_res[
                    'num_fillable']
                precs[sel_name] = prec
            else:
                precs[sel_name] = 0

        recalls = {}
        for sel_name, tp in self.overall_res['scores'].items():
            if self.overall_res['num_fill_attempts'][sel_name] > 0:
                recall = self.overall_res['tp'][sel_name] / self.overall_res[
                    'num_fill_attempts']
                recalls[sel_name] = recall
            else:
                recalls[sel_name] = 0

        for n in precs.keys():
            p = precs[n]
            r = recalls[n]
            self.overall_res['scores'][n]['precision'] = p
            self.overall_res['scores'][n]['recall'] = r

            if p + r > 0:
                self.overall_res['scores'][n]['f1'] = 2 * p * r / (p + r)
            else:
                self.overall_res['scores'][n]['f1'] = 0

        if self.out_dir is not None:
            with open(self.overall_path, 'w') as out:
                json.dump(self.overall_res, out, indent=2)
                out.write('\n')

        info = "Test F1s: "
        for n in self.overall_res['scores']:
            info += f"{n}: {self.overall_res['scores'][n]['f1']:.4f}"

        return self.overall_res
