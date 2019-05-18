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

        self.selectors = self.candidate_selectors()
        self.k = 5

        self.score_buffer = {}

    def create_score_group(self, group):
        if group not in self.score_buffer:
            self.score_buffer[group] = {
                'num_fillable': 0,
                'num_fill_attempts': 0,
                'num_instances': 0,
                'results': {
                    'system': {},
                    'oracle': {},
                },
            }

        for c in self.cutoffs:
            self.score_buffer[group]['results']['system'][f'p@{c}'] = 0
            self.score_buffer[group]['results']['system'][f'r@{c}'] = 0
            self.score_buffer[group]['results']['system']['tp'] = 0

            self.score_buffer[group]['results']['oracle'][f'p@{c}'] = 0
            self.score_buffer[group]['results']['oracle'][f'r@{c}'] = 0
            self.score_buffer[group]['results']['oracle']['tp'] = 0

    def compute_scores(self, raw_scores_labels, score_group):
        this_res = {
            'system': {},
            'oracle': {},
        }

        gold_ranks = []
        for r, (score, label) in enumerate(raw_scores_labels):
            rank = r + 1
            if label == 1:
                gold_ranks.append(rank)

        num_gold = sum([l for (_, l) in raw_scores_labels])

        for c in self.cutoffs:
            if num_gold == 0:
                p = 0
                r = 0
                gold_p = 0
                gold_r = 0
            else:
                tp_at_c = sum(
                    [1 if v <= c else 0 for v in gold_ranks]
                )
                p = 1.0 * tp_at_c / c
                r = 1.0 * tp_at_c / num_gold

                gold_tp = min(num_gold, c)
                gold_p = 1.0 * gold_tp / c
                gold_r = gold_tp / num_gold

            this_res['system'][f'p@{c}'] = p
            this_res['system'][f'r@{c}'] = r

            this_res['oracle'][f'p@{c}'] = gold_p
            this_res['oracle'][f'r@{c}'] = gold_r

            score_group['system'][f'p@{c}'] += p
            score_group['system'][f'r@{c}'] += r

            score_group['oracle'][f'p@{c}'] += gold_p
            score_group['oracle'][f'r@{c}'] += gold_r

        if raw_scores_labels[0][1] == 1:
            score_group['system']['tp'] += 1

        if num_gold > 0:
            score_group['oracle']['tp'] += 1

        return this_res

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
        def neighbor_selector(meta):
            if 0 <= meta['distance_to_event'] <= 2:
                return 'neighbor'
            else:
                return False

        def gold_candidate_selector(meta):
            if meta['source'] == 'gold':
                return 'gold'
            else:
                return False

        def neighbor_gold_selector(meta):
            if neighbor_selector(meta) and gold_candidate_selector(meta):
                return 'gold_neighbor'
            return False

        def all_selector(meta):
            return True

        def predicate_selector(meta):
            return meta['predicate']

        # return {
        #     'neighbor': neighbor_selector,
        #     'gold_candidate': gold_candidate_selector,
        #     'neighbor_gold': neighbor_gold_selector,
        #     'all': all_selector,
        #     'predicate': predicate_selector
        # }
        return [
            neighbor_selector,
            gold_candidate_selector,
            neighbor_gold_selector,
            all_selector,
            predicate_selector
        ]

    def add_result(self, doc_id, event_idx, slot_idx, score_labels, ins_meta,
                   c_meta):
        self.results.append(
            (doc_id, event_idx, slot_idx, score_labels, ins_meta, c_meta,)
        )

        data = {
            'doc_id': doc_id,
            'results': {},
            'predictions': [],
        }

        sorted_result = sorted(zip(score_labels, c_meta), reverse=True,
                               key=itemgetter(0))

        ranked_predictions = [(s, l, meta['entity']) for (s, l), meta in
                              sorted_result]

        selected_groups = {}
        for selector in self.selectors:
            for sl, meta in sorted_result:
                selection = selector(meta)
                if selection:
                    if selection not in selected_groups:
                        selected_groups[selection] = {
                            'score_labels': [],
                            'metas': []
                        }

                    meta['predicate'] = ins_meta['predicate'][1]
                    selected_groups[selection]['score_labels'].append(sl)
                    selected_groups[selection]['metas'].append(meta)

        instance_res = {
            'event_index': event_idx,
            'predicate': ins_meta['predicate'],
            'slot_index': slot_idx,
            'gold_entity': ins_meta['gold_entity'],
            'slot_name': self.slot_names[slot_idx],
            'categorized_result': {},
        }

        for group_name, members in selected_groups.items():
            self.create_score_group(group_name)
            self.score_buffer[group_name]['num_instances'] += 1
            ins_scores = self.compute_scores(
                members['score_labels'],
                self.score_buffer[group_name]['results'])
            if ins_meta['has_true']:
                self.score_buffer[group_name]['num_fillable'] += 1

            top_responses = []
            for i in range(2):
                top_responses.append(
                    (
                        members['metas'][0]['entity'],
                        members['score_labels'][i][0],
                        members['score_labels'][i][1],
                    )
                )

            if not top_responses[0][0] == ghost_entity_text:
                self.score_buffer[group_name]['num_fill_attempts'] += 1

            instance_res['categorized_result'][group_name] = {
                'scores': ins_scores,
                'top_responses': top_responses,
            }

            print(members)

            input(f'this is a member of {group_name}')

        data['results'] = instance_res
        data['predictions'] = ranked_predictions

        if self.out_dir:
            mode = 'a' if os.path.exists(self.detail_path) else 'w'
            with open(self.detail_path, mode) as res_out:
                json.dump(data, res_out, indent=2)
                res_out.write('\n')

    def collect(self):
        for group_name, member_scores in self.score_buffer.items():
            num_res = member_scores['num_fill_attempts']
            num_gold = member_scores['num_fillable']
            # for member_name, member_scores in group_scores['results'].items():
            for k in member_scores['results']['system']:
                if '@' in k:
                    member_scores['results']['system'][
                        k] /= member_scores['num_instances']

            tp = member_scores['results']['system']['tp']
            prec = tp / num_res if num_res > 0 else 0
            recall = tp / num_gold if num_gold > 0 else 0
            f1 = 2*prec*recall/(prec + recall) if prec + recall > 0 else 0

            member_scores['results']['system']['precision'] = prec
            member_scores['results']['system']['recall'] = recall
            member_scores['results']['system']['F1'] = f1

            for k in member_scores['results']['oracle']:
                if '@' in k:
                    member_scores['results']['oracle'][
                        k] /= member_scores['num_instances']

            otp = member_scores['results']['oracle']['tp']
            o_prec = otp / num_res if num_res > 0 else 0
            o_recall = otp / num_gold if num_gold > 0 else 0
            o_f1 = 2*prec*recall/(prec + recall) if prec + recall > 0 else 0

            member_scores['results']['oracle']['precision'] = o_prec
            member_scores['results']['oracle']['recall'] = o_recall
            member_scores['results']['oracle']['F1'] = o_f1

        if self.out_dir is not None:
            with open(self.overall_path, 'w') as out:
                json.dump(self.score_buffer, out, indent=2)
                out.write('\n')
