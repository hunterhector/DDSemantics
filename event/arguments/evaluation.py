import json
import os
from itertools import groupby
from operator import itemgetter
import logging
from event import util
import pdb
from pprint import pprint
from collections import Counter, defaultdict

from event.arguments.cloze_readers import ghost_entity_text
from event.io.dataset.utils import nombank_pred_text


class ImplicitEval:
    def __init__(self, slot_names, out_dir=None):
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

        self.overall_results = {}

    def create_score_group(self, group):
        if group not in self.overall_results:
            self.overall_results[group] = {
                'num_fillable': 0,
                'num_fill_attempts': 0,
                'num_instances': 0,
                'results': {
                    'system': {},
                    'oracle': {},
                },
            }

            for c in self.cutoffs:
                self.overall_results[group]['results']['system'][f'p@{c}'] = 0
                self.overall_results[group]['results']['system']['tp'] = 0

                self.overall_results[group]['results']['oracle'][f'p@{c}'] = 0
                self.overall_results[group]['results']['oracle']['tp'] = 0

    @staticmethod
    def max_dice(entity, answers):
        max_dice = 0
        e_span = entity['span']
        for answer in answers:
            a_span = answer['span']
            span_set = set(range(e_span[0], e_span[1]))
            ans_span_set = set(range(a_span[0], a_span[1]))
            inter = len(span_set.intersection(ans_span_set))
            dice = 2 * inter / (len(span_set) + len(ans_span_set))

            if dice > max_dice:
                max_dice = dice
        return max_dice

    def compute_scores(self, ranked_cands, answers, score_group):
        this_res = {
            'system': {},
            'oracle': {},
        }

        ranked_entities = ranked_cands['metas']

        gold_ranks = []
        dices = []
        for r, ent in enumerate(ranked_entities):
            rank = r + 1
            d_score = self.max_dice(ent, answers)
            dices.append(d_score)
            if d_score > 0:
                gold_ranks.append(rank)

        for c in self.cutoffs:
            # Scaled tp@c.
            tp_at_c = sum(dices[:c])
            p = 1.0 * tp_at_c / c

            this_res['system'][f'p@{c}'] = p
            score_group['system'][f'p@{c}'] += p

        if len(gold_ranks) > 0:
            this_res['oracle']['p@1'] = 1
            score_group['oracle']['p@1'] += 1

        score_group['system']['tp'] += dices[0]

        if len(gold_ranks) > 0:
            score_group['oracle']['tp'] += 1

        return this_res

    def add_prediction(self, doc_id, event_indexes, slot_indexes, coh_scores,
                       candidate_meta, instance_meta):
        for (((event_idx, slot_idx), result), ins_meta) in zip(groupby(
                zip(zip(
                    event_indexes, slot_indexes), coh_scores, candidate_meta),
                key=itemgetter(0)), instance_meta):
            _, raw_scores, c_meta = zip(*result)

            self.add_result(
                doc_id, event_idx, slot_idx, raw_scores, ins_meta, c_meta
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
            return "all"

        def predicate_selector(meta):
            return nombank_pred_text(meta['predicate'])

        return [
            neighbor_selector,
            gold_candidate_selector,
            neighbor_gold_selector,
            all_selector,
            predicate_selector
        ]

    def add_result(self, doc_id, event_idx, slot_idx, raw_scores, ins_meta,
                   c_meta):
        data = {
            'doc_id': doc_id,
            'results': {},
            'predictions': [],
        }

        sorted_result = sorted(zip(raw_scores, c_meta), reverse=True,
                               key=itemgetter(0))

        ranked_predictions = [(s, meta['entity']) for s, meta in sorted_result]

        selected_groups = {}
        for selector in self.selectors:
            for s, meta in sorted_result:
                selection = selector(meta)
                if selection:
                    if selection not in selected_groups:
                        selected_groups[selection] = {
                            'scores': [],
                            'metas': []
                        }

                    meta['predicate'] = ins_meta['predicate']
                    selected_groups[selection]['scores'].append(s)
                    selected_groups[selection]['metas'].append(meta)

        instance_res = {
            'event_index': event_idx,
            'predicate': ins_meta['predicate'],
            'slot_name': self.slot_names[slot_idx],
            'slot_index': slot_idx,
            'answers': ins_meta['answers'],
            'categorized_result': {},
        }

        for group_name, members in selected_groups.items():
            self.create_score_group(group_name)
            self.overall_results[group_name]['num_instances'] += 1
            ins_scores = self.compute_scores(
                members,
                ins_meta['answers'],
                self.overall_results[group_name]['results']
            )
            if len(ins_meta['answers']) > 0:
                self.overall_results[group_name]['num_fillable'] += 1

            top_responses = []
            topk = min(2, len(members['scores']))
            for i in range(topk):
                top_responses.append(
                    (
                        members['metas'][0]['entity'],
                        members['scores'][i],
                    )
                )

            if not top_responses[0][0] == ghost_entity_text:
                self.overall_results[group_name]['num_fill_attempts'] += 1

            instance_res['categorized_result'][group_name] = {
                'scores': ins_scores,
                'top_responses': top_responses,
            }

        data['results'] = instance_res
        data['predictions'] = ranked_predictions

        if self.out_dir:
            mode = 'a' if os.path.exists(self.detail_path) else 'w'
            with open(self.detail_path, mode) as res_out:
                json.dump(data, res_out, indent=2)
                res_out.write('\n')

    def collect(self):
        for group_name, member_scores in self.overall_results.items():
            num_res = member_scores['num_fill_attempts']
            num_gold = member_scores['num_fillable']

            for k in member_scores['results']['system']:
                if '@' in k:
                    member_scores['results']['system'][
                        k] /= member_scores['num_instances']

            tp = member_scores['results']['system']['tp']
            prec = tp / num_res if num_res > 0 else 0
            recall = tp / num_gold if num_gold > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0

            member_scores['results']['system']['precision'] = prec
            member_scores['results']['system']['recall'] = recall
            member_scores['results']['system']['F1'] = f1

            for k in member_scores['results']['oracle']:
                if '@' in k:
                    member_scores['results']['oracle'][
                        k] /= member_scores['num_instances']

            tp = member_scores['results']['oracle']['tp']
            prec = tp / num_res if num_res > 0 else 0
            recall = tp / num_gold if num_gold > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0

            member_scores['results']['oracle']['precision'] = prec
            member_scores['results']['oracle']['recall'] = recall
            member_scores['results']['oracle']['F1'] = f1

        if self.out_dir is not None:
            with open(self.overall_path, 'w') as out:
                json.dump(self.overall_results, out, indent=2)
                out.write('\n')
