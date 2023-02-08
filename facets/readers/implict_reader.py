"""
Process implicit argument datasets.
"""
import json
import re
from typing import Any, Iterator

from forte.data import DataPack
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from forte.data.caster import MultiPackBoxer
from ft.onto.base_ontology import Sentence

from event.io.dataset.base import Corpus, DEDocument
from event.io.dataset.nombank import NomBank
from onto.facets import EventMention, EntityMention, EventArgument


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
            "wsj_file_pattern": "\d\d/wsj_.*\.mrg",
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
        sentence_spans = []
        for tokenized_sentence in tokenized_text:
            sent_start = len(doc_text)
            for raw_token in tokenized_sentence:
                doc_text += splitter + raw_token
                token_spans.append((len(doc_text) - len(raw_token), len(doc_text)))
                splitter = " "
            sentence_spans.append((sent_start, len(doc_text)))
            splitter = "\n"

        pack.set_text(doc_text)

        for b, e in sentence_spans:
            pack.add_entry(Sentence(pack, b, e))

        # store event mentions by their span
        span2evt = {}
        for b, e, types in evt_triggers:
            # Convert from token span to character span.
            mention = EventMention(pack, token_spans[b][0], token_spans[e][1])
            for type_name, score in types:
                mention.types.append(type_name)
            pack.add_entry(mention)
            span2evt[(b, e)] = mention

        # store entity mentions by their span
        span2ent = {}
        for b, e, arg_types in ent_spans:
            mention = EntityMention(pack, token_spans[b][0], token_spans[e][1])
            pack.add_entry(mention)
            span2ent[(b, e)] = mention

        # use the spans to link entity mentions and event mentions
        for evt_span, ent_span, arg_type in gold_evt_links:
            ent_mention = span2ent[tuple(ent_span)]
            evt_mention = span2evt[tuple(evt_span)]
            argument = EventArgument(pack, evt_mention, ent_mention)
            argument.role = arg_type

            match = next(re.finditer(r"[a-z]+\d+([a-z]+\d+)([a-z]+)", arg_type))

            argument.pb_role = match.group(1)
            argument.vb_role = match.group(2)
            pack.add_entry(argument)

        yield pack

    @classmethod
    def default_configs(cls):
        return {
            # https://nlp.jhu.edu/rams/
            "rams_json_path": "../data/RAMS_1.0/data/train.jsonlines",
        }
