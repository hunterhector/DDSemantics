from forte.data import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence


class ImplicitStats(PackProcessor):
    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            pass
