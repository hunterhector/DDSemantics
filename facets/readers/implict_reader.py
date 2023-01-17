"""
Process implicit argument datasets.
"""
import os.path
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
            for line in rams_file:
                root = json.loads(line)
                yield (
                    root["doc_key"],
                    root["ent_spans"],
                    root["evt_triggers"],
                    root["sentences"],
                    root["gold_evt_links"]
                )

    def _parse_pack(self, collection: Any) -> Iterator[PackType]:
        doc_name, ent_spans, evt_triggers, tokenized_text, gold_evt_links = collection
        pack = DataPack(doc_name)

        doc_text = ""
        splitter = ""
        token_spans = []
        for tokenized_sentence in tokenized_text:
            for raw_token in tokenized_sentence:
                doc_text += splitter + raw_token
                token_spans.append((len(doc_text) - len(raw_token), len(doc_text)))
                splitter = " "
            splitter = "\n"

        pack.set_text(doc_text)
        print(pack.text)
        print(evt_triggers)

        raise Exception

    @classmethod
    def default_configs(cls):
        return {
            # https://nlp.jhu.edu/rams/
            "rams_json_path": "../data/RAMS_1.0/data/train.jsonlines",
        }
