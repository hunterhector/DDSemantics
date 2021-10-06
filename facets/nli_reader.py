from typing import Any, Iterator
from operator import itemgetter

from datasets import load_dataset

from forte.data import DataPack
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from onto.facets import Premise, Hypothesis, NLIPair


class MultiNLIReader(PackReader):
    def __init__(self):
        super().__init__()
        self._keys = ["pairID", "premise", "hypothesis", "label"]

    def _collect(self) -> Iterator[Any]:
        nli_dataset = load_dataset("multi_nli")

        key_indexes = range(len(self._keys))

        items = itemgetter(*key_indexes)(
            itemgetter(*self._keys)(
                nli_dataset["train"]
            )
        )

        for idx in range(len(items[0])):
            yield [item[idx] for item in itemgetter(*key_indexes)(items)]

    def _parse_pack(self, nli_instance) -> Iterator[PackType]:
        pair_id, source, target, label = nli_instance

        pack = DataPack(pair_id)
        text = source + "\n" + target + "\n"
        pack.set_text(text)

        premise = Premise(pack, 0, len(source))
        hypo = Hypothesis(pack, len(source) + 1, len(text) - 1)

        pair = NLIPair(pack)
        pair.set_parent(premise)
        pair.set_child(hypo)

        pair.entailment = {
            "entailment": 0,
            "neutral": 0,
            "contradiction": 0,
        }

        if label == 2:
            pair.entailment["contradiction"] = 1
        elif label == 0:
            pair.entailment["entailment"] = 1
        elif label == 1:
            pair.entailment["neutral"] = 1
        else:
            raise ValueError("Unknown label value.")

        yield pack
