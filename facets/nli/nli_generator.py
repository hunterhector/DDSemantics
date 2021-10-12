from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import MultiPack, DataPack
from forte.processors.base import MultiPackProcessor, PackProcessor
from ft.onto.base_ontology import Dependency
from onto.facets import NLIPair, Premise, Hypothesis


def create_nli(pack: DataPack, premise_text, hypothesis_text):
    text = premise_text + "\n" + hypothesis_text + "\n"
    pack.set_text(text)

    premise = Premise(pack, 0, len(premise_text))
    hypo = Hypothesis(pack, len(premise_text) + 1, len(text) - 1)

    pair = NLIPair(pack)
    pair.set_parent(premise)
    pair.set_child(hypo)


class TweakData(MultiPackProcessor):
    def tweak_nli_text(self, premise: Premise, hypo: Hypothesis):
        pre_deps = list(premise.get(Dependency))
        hypo_deps = list(premise.get(Dependency))

        return [(premise.text, hypo.text)]

    def _process(self, input_pack: MultiPack):
        src_pack = input_pack.get_pack(self.configs.source_pack_name)

        instance: NLIPair
        for instance in src_pack.get(NLIPair):
            premise = instance.get_parent()
            hypo = instance.get_child()

            for i, (new_prem, new_hypo) in enumerate(
                    self.tweak_nli_text(premise, hypo)):
                pack = input_pack.add_pack(
                    f"generated_{i}", input_pack.pack_name + f"_{i}")
                create_nli(pack, new_prem, new_hypo)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "source_pack_name": "default",
            "new_pack_prefix": "generated_"
        }


class NLIProcessor(PackProcessor):
    def __init__(self):
        super().__init__()
        self.__id2label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.device = "cpu" if not torch.cuda.is_available() else \
            self.configs.device

        self.tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            configs.model_name).to(self.device)

        self.is_bart = False
        if configs.model_name.split("/")[1].split("-")[0] == "bart":
            self.is_bart = True

    def _process(self, input_pack: DataPack):
        instance: NLIPair
        for instance in input_pack.get(NLIPair):
            premise = instance.get_parent().text
            hypo = instance.get_child().text
            results = self._nli_inference(premise, hypo)

            for k, v in enumerate(results):
                instance.entailment[self.__id2label[k]] = v

    def _nli_inference(self, premise: str, hypothesis: str):
        input_pair = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            return_token_type_ids=True,
            truncation=True
        )

        input_ids = torch.Tensor(
            input_pair['input_ids']).long().unsqueeze(0).to(self.device)

        token_type_ids = None
        if not self.is_bart:
            token_type_ids = torch.Tensor(
                input_pair['token_type_ids']
            ).long().unsqueeze(0).to(self.device)

        attention_mask = torch.Tensor(
            input_pair['attention_mask']).long().unsqueeze(0).to(self.device)

        if self.is_bart:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=None
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )

        return torch.softmax(outputs[0], dim=1)[0].cpu().tolist()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "model_name": "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            "device": "cuda:0"
        }
