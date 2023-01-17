"""
Process implicit argument datasets.
"""
from typing import Any, Iterator
import json

from forte.data import DataPack, Span
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from forte.processors.writers import PackIdJsonPackWriter

from event.io.dataset.nombank import NomBank
from event.io.dataset.base import Corpus, DEDocument


class GCNombankReader(PackReader):
    def _collect(self) -> Iterator[Any]:
        corpus = Corpus()
        self.nom_parser = NomBank(self.configs, corpus)
        yield from self.nom_parser

    def _parse_pack(self, doc: DEDocument) -> Iterator[PackType]:
        datapack = DataPack()
        datapack.set_text(doc.doc_text)

    @classmethod
    def default_configs(cls):
        return {
            # https://sled.eecs.umich.edu/post/resources/nominal-semantic-role-labeling/
            "implicit_path": "~/Documents/projects/data/implicit",
            # https://nlp.cs.nyu.edu/meyers/nombank/nombank.1.0.zip
            "nombank_path": "~/Documents/projects/data/nombank",
            "wsj_path": "~/Document/projects/data/wsj",
        }


class RamsReader(PackReader):

    def _collect(self) -> Iterator[Any]:
        with open(self.configs.rams_json_path) as rams_file:
            root = json.load(rams_file)
            root["ent_spans"]

    def _parse_pack(self, collection: Any) -> Iterator[PackType]:
        pass

    @classmethod
    def default_configs(cls):
        return {
            # https://nlp.jhu.edu/rams/
            "rams_json_path": "~/Documents/projects/data/RAMS_1.0/data/train.jsonlines",
        }