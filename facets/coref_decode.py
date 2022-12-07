from typing import Dict

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence, EventMention, Token

from facets.coref_data import bracket_mention
from facets.common.event_utils import all_valid_events, events2sentences
from onto.facets import Hopper


class RemoveHopper(PackProcessor):
    def _process(self, input_pack: DataPack):
        hoppers = list(input_pack.get(Hopper))
        for h in hoppers:
            input_pack.delete_entry(h)


class BasePairwiseCoref(PackProcessor):
    """
    Create coreference cluster with pairwise model.
    """

    def _process(self, input_pack: DataPack):
        all_events = all_valid_events(input_pack)

        coref_chains = []
        idx2chains: Dict[int, int] = {}

        # Iterate every mention to find antecedent (except first one).
        for i in range(1, len(all_events)):
            # Scan backwards.
            for p in range(i - 1, 0, -1):
                if self._pair_predict(all_events[p], all_events[i]):
                    if p in idx2chains:
                        coref_chains[idx2chains[p]].append(i)
                        idx2chains[i] = idx2chains[p]
                    else:
                        coref_chains.append([p, i])
                        idx2chains[p] = len(coref_chains) - 1
                        idx2chains[i] = len(coref_chains) - 1
                    break

        for chain in coref_chains:
            hopper = Hopper(input_pack)
            for evm_index in chain:
                hopper.add_member(all_events[evm_index])

    def _pair_predict(
            self,
            evm1: EventMention,
            evm2: EventMention,
    ) -> bool:
        raise NotImplementedError


class SameTokenCoref(BasePairwiseCoref):
    def _pair_predict(
            self,
            evm1: EventMention,
            evm2: EventMention,
    ) -> bool:
        head1 = ""
        for word1 in evm1.get(Token):
            head1 = word1.lemma.lower()

        head2 = ""
        for word2 in evm2.get(Token):
            head2 = word2.lemma.lower()

        return head1 == head2


class PairwiseCoref(BasePairwiseCoref):
    """
    Create coreference cluster with pairwise model.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        # self._model = AutoModel.from_pretrained(
        #     self.configs.model_path
        # )
        # self._tokenizer = AutoTokenizer.from_pretrained(
        #     'distilbert-base-uncased'
        # )
        self._model = resources.get("coref_model")
        self._tokenizer = resources.get("tokenizer")

        self.events2sents: Dict[int, Sentence] = {}

    def _process(self, input_pack: DataPack):
        self.events2sents = events2sentences(input_pack)

    def _pair_predict(
            self,
            evm1: EventMention,
            evm2: EventMention,
    ) -> bool:
        content1 = bracket_mention(self.events2sents[evm1.tid], evm1)
        content2 = bracket_mention(self.events2sents[evm2.tid], evm2)

        tokens1 = self._tokenizer(content1, truncation=True, padding=True)
        tokens2 = self._tokenizer(content2, truncation=True, padding=True)

        self._model(content1, content2)

        return True
