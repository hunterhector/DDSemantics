import json
import os
from itertools import groupby
from operator import itemgetter
import logging
from event import util

from event.arguments.cloze_readers import ghost_entity_text


class ImplicitEval:
    def __init__(self, slot_names, out_dir=None):
        self.num_instances = 0
        self.results = []
        self.out_dir = out_dir
        self.cutoffs = [1, 5, 10]

        self.slot_names = slot_names

        if self.out_dir is not None:
            self.detail_path = os.path.join(self.out_dir, 'detailed_out.json')
            self.overall_path = os.path.join(self.out_dir, 'overall.json')
            if os.path.exists(self.detail_path):
                util.append_num_to_path(self.detail_path)

    def add_prediction(self, doc_id, event_indexes, slot_indexes, coh_scores,
                       gold_labels, candidate_meta, instance_meta):
        for (((event_idx, slot_idx), result), ins_meta) in zip(groupby(
                zip(zip(event_indexes, slot_indexes),
                    zip(coh_scores, gold_labels),
                    candidate_meta,
                    ),
                key=itemgetter(0)), instance_meta):
            _, score_labels, c_meta = zip(*result)
            self.add_result(
                doc_id, event_idx, slot_idx, score_labels, ins_meta, c_meta
            )

    def add_result(self, doc_id, event_idx, slot_idx, score_labels, ins_meta,
                   c_meta):
        self.results.append(
            (doc_id, event_idx, slot_idx, score_labels, ins_meta, c_meta,)
        )

    def run(self):
        overall_res = {
            'num_docs': 0,
            'num_instances': 0,
            'num_fillable': 0,
            'num_fill_attempts': 0,
            'tp': 0,
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

            for _, event_idx, slot_idx, score_labels, ins_meta, c_meta in ref:
                gold_rank = []
                top_k = []

                sorted_result = sorted(zip(score_labels, c_meta), reverse=True,
                                       key=itemgetter(0))

                top_result = None
                for r, ((score, label), meta) in enumerate(sorted_result):
                    if label == 1:
                        gold_rank.append(r + 1)
                    if r < 5:
                        top_k.append((score, label, meta['entity'], r))
                    if r == 0:
                        top_result = score, label, meta['entity']

                num_gold = len(gold_rank)

                if ins_meta['has_true']:
                    overall_res['num_fillable'] += 1

                if not top_result[2] == ghost_entity_text:
                    overall_res['num_fill_attempts'] += 1

                instance_res = {
                    'event_index': event_idx,
                    'slot_index': slot_idx,
                    'slot_name': self.slot_names[slot_idx],
                    'num_gold': num_gold,
                    'gold_ranks': gold_rank,
                    'top_k': top_k,
                    'predicate': ins_meta['predicate'],
                    'gold_entity': ins_meta['gold_entity'],
                    'scores': {},
                    'gold': {},
                }

                overall_res['num_instances'] += 1

                # If one of the gold instance is ranked as the best.
                bingo = any([r == 1 for r in gold_rank])
                if bingo:
                    overall_res['tp'] += 1

                for c in self.cutoffs:
                    if num_gold == 0:
                        p = 0
                        r = 0
                        gold_p = 0
                        gold_r = 0
                    else:
                        tp_at_c = sum([1 if v <= c else 0 for v in gold_rank])
                        p = 1.0 * tp_at_c / c
                        r = 1.0 * tp_at_c / num_gold
                        gold_tp = min(num_gold, c)
                        gold_p = 1.0 * gold_tp / c
                        gold_r = gold_tp / num_gold

                    instance_res['scores']['p@%d' % c] = p
                    instance_res['scores']['r@%d' % c] = r

                    instance_res['gold']['p@%d' % c] = gold_p
                    instance_res['gold']['r@%d' % c] = gold_r

                    overall_res['scores']['p@%d' % c] += p
                    overall_res['scores']['r@%d' % c] += r

                    overall_res['gold']['p@%d' % c] += gold_p
                    overall_res['gold']['r@%d' % c] += gold_r

                data['predictions'].append(instance_res)

            if self.out_dir:
                mode = 'a' if os.path.exists(self.detail_path) else 'w'
                with open(self.detail_path, mode) as res_out:
                    json.dump(data, res_out, indent=2)
                    res_out.write('\n')

        for k, v in overall_res['scores'].items():
            overall_res['scores'][k] = v / overall_res['num_instances']

        for k, v in overall_res['gold'].items():
            overall_res['gold'][k] = v / overall_res['num_instances']

        prec = overall_res['tp'] / overall_res['num_fillable']
        recall = overall_res['tp'] / overall_res['num_fill_attempts']

        overall_res['scores']['precision'] = prec
        overall_res['scores']['recall'] = recall

        if prec + recall > 0:
            overall_res['scores']['f1'] = 2 * prec * recall / (prec + recall)
        else:
            overall_res['scores']['f1'] = 0

        if self.out_dir is not None:
            with open(self.overall_path, 'w') as out:
                json.dump(overall_res, out, indent=2)
                out.write('\n')
        else:
            logging.info("Test p@1 is %.4f." % overall_res['scores']['p@1'])
            logging.info("Test F1 is %.4f." % overall_res['scores']['f1'])

        return overall_res
