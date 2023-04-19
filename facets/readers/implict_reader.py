"""
Process implicit argument datasets.
"""
import json
import re
from typing import Any, Iterator, Dict

from forte.data import DataPack
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Sentence

from event.io.dataset.base import Corpus, DEDocument
from event.io.dataset.nombank import NomBank
from onto.facets import EventMention, EntityMention, EventArgument


class GCNombankReader(PackReader):
    def _collect(self) -> Iterator[Any]:
        with open(self.configs.bnb_json_path) as bnb_file:
            for line in bnb_file:
                yield json.loads(line)

    def _parse_pack(self, bnb_data: Dict) -> Iterator[PackType]:
        datapack = DataPack(bnb_data["doc_key"])
        doc_text, token_spans, sentence_spans = construct_text(bnb_data, "sentences")
        datapack.set_text(doc_text)

        yield datapack

    @classmethod
    def default_configs(cls):
        return {
            # https://github.com/pitrack/arglinking/tree/master/data/bnb
            "bnb_json_path": None
        }


def construct_text(raw_json: Dict, text_key: str):
    doc_text = ""

    splitter = ""
    token_spans = []
    sentence_spans = []
    for tokenized_sentence in raw_json[text_key]:
        sent_start = len(doc_text)
        for raw_token in tokenized_sentence:
            doc_text += splitter + raw_token
            token_spans.append((len(doc_text) - len(raw_token), len(doc_text)))
            splitter = " "
        sentence_spans.append((sent_start, len(doc_text)))
        splitter = "\n"

    return doc_text, token_spans, sentence_spans


class GVDBReader(PackReader):

    def _collect(self) -> Iterator[Any]:
        with open(self.configs.gvdb_json_path) as gvdb_file:
            for line in gvdb_file:
                yield json.loads(line)

    def _parse_pack(self, collection: Any) -> Iterator[PackType]:
        pack = DataPack(collection["doc_key"])

        doc_text, token_spans, sentence_spans = construct_text(collection, "full_text")

        # doc_text = ""
        #
        # splitter = ""
        # token_spans = []
        # sentence_spans = []
        # for tokenized_sentence in collection["full_text"]:
        #     sent_start = len(doc_text)
        #     for raw_token in tokenized_sentence:
        #         doc_text += splitter + raw_token
        #         token_spans.append((len(doc_text) - len(raw_token), len(doc_text)))
        #         splitter = " "
        #     sentence_spans.append((sent_start, len(doc_text)))
        #     splitter = "\n"

        pack.set_text(doc_text)

        for b, e in sentence_spans:
            pack.add_entry(Sentence(pack, b, e))

        # In this GVDB setting, we assume one event per doc, so let's just use the
        # first sentence as the event mention span.
        event = EventMention(pack, sentence_spans[0][0], sentence_spans[0][1])
        pack.add_entry(event)

        # print(f"{len(token_spans)} tokens in the article")
        for token_begin, token_end, role, text, token_text in collection["spans"]:
            # print(f"Span is {token_begin}, {token_end}")

            try:
                char_begin = token_spans[token_begin][0]
                char_end = token_spans[token_end - 1][1]
            except IndexError:
                print(f"Processing {pack.pack_id}")
                import pdb;
                pdb.set_trace()

            entity = EntityMention(pack, char_begin, char_end)
            pack.add_entry(entity)

            argument = EventArgument(pack, event, entity)
            argument.role = role
            pack.add_entry(argument)

        yield pack

    @classmethod
    def default_configs(cls):
        return {
            "gvdb_json_path": None,
        }


class RamsReader(PackReader):

    def _collect(self) -> Iterator[Any]:
        with open(self.configs.rams_json_path) as rams_file:
            for line in rams_file:
                yield json.loads(line)
                # yield (
                #     root["doc_key"],
                #     root["ent_spans"],
                #     root["evt_triggers"],
                #     root["sentences"],
                #     root["gold_evt_links"]
                # )

    def _parse_pack(self, root: Any) -> Iterator[PackType]:
        # doc_name, ent_spans, evt_triggers, tokenized_text, gold_evt_links = collection
        pack = DataPack(root["doc_key"])

        doc_text, token_spans, sentence_spans = construct_text(root, "sentences")

        # doc_text = ""
        #
        # splitter = ""
        # token_spans = []
        # sentence_spans = []
        # for tokenized_sentence in tokenized_text:
        #     sent_start = len(doc_text)
        #     for raw_token in tokenized_sentence:
        #         doc_text += splitter + raw_token
        #         token_spans.append((len(doc_text) - len(raw_token), len(doc_text)))
        #         splitter = " "
        #     sentence_spans.append((sent_start, len(doc_text)))
        #     splitter = "\n"

        pack.set_text(doc_text)

        for b, e in sentence_spans:
            pack.add_entry(Sentence(pack, b, e))

        # store event mentions by their span
        span2evt = {}
        for b, e, types in root["evt_triggers"]:
            # Convert from token span to character span.
            mention = EventMention(pack, token_spans[b][0], token_spans[e][1])
            for type_name, score in types:
                mention.types.append(type_name)
            pack.add_entry(mention)
            span2evt[(b, e)] = mention

        # store entity mentions by their span
        span2ent = {}
        for b, e, arg_types in root["ent_spans"]:
            mention = EntityMention(pack, token_spans[b][0], token_spans[e][1])
            pack.add_entry(mention)
            span2ent[(b, e)] = mention

        # use the spans to link entity mentions and event mentions
        for evt_span, ent_span, arg_type in root["gold_evt_links"]:
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
