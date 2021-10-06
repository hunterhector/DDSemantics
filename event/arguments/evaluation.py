import json
import os
import pdb
from operator import itemgetter
from event import util

from event.arguments.data.cloze_readers import ghost_entity_text
from event.io.dataset.utils import normalize_pred_text


def save_div(a, b):
    return a / b if b > 0 else 0


class ImplicitEval:
    def __init__(self, out_dir=None):
        self.out_dir = out_dir
        self.cutoffs = [1, 5, 10]

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            self.detail_path = os.path.join(self.out_dir, "detailed_out.json")
            self.overall_path = os.path.join(self.out_dir, "overall.json")

            if os.path.exists(self.overall_path):
                util.append_num_to_path(self.overall_path)

            if os.path.exists(self.detail_path):
                util.append_num_to_path(self.detail_path)

        self.selectors = self.candidate_selectors()
        self.k = 5

        self.overall_results = {}

    def create_score_group(self, group_type, group_name):
        if group_type not in self.overall_results:
            self.overall_results[group_type] = {}

        if group_name not in self.overall_results[group_type]:
            self.overall_results[group_type][group_name] = {
                "num_fillable": 0,
                "num_fill_attempts": 0,
                "num_instances": 0,
                "results": {
                    "system": {},
                    "oracle": {},
                },
            }

        for c in self.cutoffs:
            self.overall_results[group_type][group_name]["results"]["system"][
                f"p@{c}"
            ] = 0
            self.overall_results[group_type][group_name]["results"]["system"]["tp"] = 0

            self.overall_results[group_type][group_name]["results"]["oracle"][
                f"p@{c}"
            ] = 0
            self.overall_results[group_type][group_name]["results"]["oracle"]["tp"] = 0

    @staticmethod
    def max_dice(entity, answers):
        max_dice = 0
        e_span = entity["span"]
        for answer in answers:
            a_span = answer["span"]
            span_set = set(range(e_span[0], e_span[1]))
            ans_span_set = set(range(a_span[0], a_span[1]))
            inter = len(span_set.intersection(ans_span_set))
            dice = 2 * inter / (len(span_set) + len(ans_span_set))

            if dice > max_dice:
                max_dice = dice
        return max_dice

    def compute_scores(self, ranked_cands, answers, score_group):
        this_res = {
            "system": {},
            "oracle": {},
        }

        ranked_entities = ranked_cands["metas"]

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

            this_res["system"][f"p@{c}"] = p
            score_group["system"][f"p@{c}"] += p

        if len(gold_ranks) > 0:
            this_res["oracle"]["p@1"] = 1
            score_group["oracle"]["p@1"] += 1

        score_group["system"]["tp"] += dices[0]

        if len(gold_ranks) > 0:
            score_group["oracle"]["tp"] += 1

        return this_res

    def add_prediction(self, coh_scores, metadata):
        for l_candidate_meta, instance_meta in zip(
            metadata["candidate"], metadata["instance"]
        ):
            self.add_result(instance_meta, l_candidate_meta, coh_scores)

    @staticmethod
    def candidate_selectors():
        def neighbor_selector(meta):
            if 0 <= meta["distance_to_event"] <= 2:
                return "basic", "neighbor"
            else:
                return False

        def gold_candidate_selector(meta):
            if meta["source"] == "gold":
                return "basic", "gold"
            else:
                return False

        def neighbor_gold_selector(meta):
            if neighbor_selector(meta) and gold_candidate_selector(meta):
                return "basic", "gold_neighbor"
            return False

        def all_selector(meta):
            return "basic", "all"

        def predicate_selector(meta):
            return "predicate", normalize_pred_text(meta["predicate"])

        return [
            neighbor_selector,
            gold_candidate_selector,
            neighbor_gold_selector,
            all_selector,
            predicate_selector,
        ]

    def add_result(self, ins_meta, c_meta, raw_scores):
        data = {
            "doc_id": ins_meta["docid"],
            "results": {},
            "predictions": [],
        }

        sorted_result = sorted(zip(raw_scores, c_meta), reverse=True, key=itemgetter(0))

        ranked_predictions = [(s, meta["entity"]) for s, meta in sorted_result]

        selected_groups = {}
        for selector in self.selectors:
            for s, meta in sorted_result:
                selection = selector(meta)
                if selection:
                    selector_type, select_result = selection
                    if selector_type not in selected_groups:
                        selected_groups[selector_type] = {}

                    if select_result not in selected_groups[selector_type]:
                        selected_groups[selector_type][select_result] = {
                            "scores": [],
                            "metas": [],
                        }

                    meta["predicate"] = ins_meta["predicate"]
                    selected_groups[selector_type][select_result]["scores"].append(s)
                    selected_groups[selector_type][select_result]["metas"].append(meta)

        instance_res = {
            "predicate": ins_meta["predicate"],
            "answers": ins_meta["answers"],
            "results": {},
        }

        for group_type, groups in selected_groups.items():
            for group_name, members in groups.items():
                self.create_score_group(group_type, group_name)
                result_holder = self.overall_results[group_type][group_name]

                result_holder["num_instances"] += 1
                ins_scores = self.compute_scores(
                    members, ins_meta["answers"], result_holder["results"]
                )
                if len(ins_meta["answers"]) > 0:
                    result_holder["num_fillable"] += 1

                top_responses = []
                topk = min(2, len(members["scores"]))
                for i in range(topk):
                    top_responses.append(
                        (
                            members["metas"][0]["entity"],
                            members["scores"][i],
                        )
                    )

                if not top_responses[0][0] == ghost_entity_text:
                    result_holder["num_fill_attempts"] += 1

                if group_type not in instance_res["results"]:
                    instance_res["results"][group_type] = {}

                instance_res["results"][group_type][group_name] = {
                    "scores": ins_scores,
                    "top_responses": top_responses,
                }

        data["results"] = instance_res
        data["predictions"] = ranked_predictions

        if self.out_dir:
            mode = "a" if os.path.exists(self.detail_path) else "w"
            with open(self.detail_path, mode) as res_out:
                # Dump each result in one line.
                json.dump(data, res_out)
                res_out.write("\n")

    def collect(self):
        # TODO: P@N seems to be wrong.
        for group_type, groups in self.overall_results.items():
            for group_name, member_scores in groups.items():
                num_res = member_scores["num_fill_attempts"]
                num_gold = member_scores["num_fillable"]

                for k in member_scores["results"]["system"]:
                    if "@" in k:
                        member_scores["results"]["system"][k] /= member_scores[
                            "num_instances"
                        ]

                tp = member_scores["results"]["system"]["tp"]
                prec = save_div(tp, num_res)
                recall = save_div(tp, num_gold)
                f1 = save_div(2 * prec * recall, (prec + recall))

                member_scores["results"]["system"]["precision"] = prec
                member_scores["results"]["system"]["recall"] = recall
                member_scores["results"]["system"]["F1"] = f1

                for k in member_scores["results"]["oracle"]:
                    if "@" in k:
                        member_scores["results"]["oracle"][k] /= member_scores[
                            "num_instances"
                        ]

                tp = member_scores["results"]["oracle"]["tp"]
                prec = save_div(tp, num_res)
                recall = save_div(tp, num_gold)
                f1 = save_div(2 * prec * recall, (prec + recall))

                member_scores["results"]["oracle"]["precision"] = prec
                member_scores["results"]["oracle"]["recall"] = recall
                member_scores["results"]["oracle"]["F1"] = f1

            if self.out_dir is not None:
                with open(self.overall_path, "w") as out:
                    json.dump(self.overall_results, out, indent=2)
                    out.write("\n")
