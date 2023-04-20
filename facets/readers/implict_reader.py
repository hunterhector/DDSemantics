"""
Process implicit argument datasets.
"""
import json
import re
from collections import defaultdict
from typing import Any, Iterator, Dict

from forte.data import DataPack
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Sentence

from onto.facets import EventMention, EntityMention, EventArgument


class GCNombankReader(PackReader):
    def _collect(self) -> Iterator[Any]:
        data_by_doc = defaultdict(list)
        with open(self.configs.bnb_json_path) as bnb_file:
            for line in bnb_file:
                data = json.loads(line)
                data_by_doc[data["doc_key"]].append(data)

        yield from data_by_doc.items()

    def _parse_pack(self, bnb_data) -> Iterator[PackType]:
        doc_key, arg_group = bnb_data
        datapack = DataPack(doc_key)

        # Text data is duplicated in these groups. So just take one group to reconstruct.
        doc_text, token_spans, sentence_spans = construct_text(arg_group[0], "sentences")
        datapack.set_text(doc_text)

        for begin, end in sentence_spans:
            datapack.add_entry(Sentence(datapack, begin, end))

        for bnb_data in arg_group:
            trigger_data = bnb_data["trigger"]
            trigger_begin, trigger_end = token_spans[trigger_data["span"][0]][0], \
                token_spans[trigger_data["span"][1]][1]
            trigger = EventMention(datapack, trigger_begin, trigger_end)
            trigger.id = trigger_data["node_id"]

            for role, arg_details in bnb_data["arguments"].items():
                for arg_detail in arg_details:
                    arg_begin, arg_end = token_spans[arg_detail["span"][0]][0], \
                        token_spans[arg_detail["span"][1]][1]
                    arg_ent = EntityMention(datapack, arg_begin, arg_end)
                    datapack.add_entry(arg_ent)

                    argument = EventArgument(datapack, trigger, arg_ent)
                    argument.role = role
                    datapack.add_entry(argument)

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

        pack.set_text(doc_text)

        for b, e in sentence_spans:
            pack.add_entry(Sentence(pack, b, e))

        # In this GVDB setting, we assume one event per doc, so let's just use the
        # first sentence as the event mention span.
        event = EventMention(pack, sentence_spans[0][0], sentence_spans[0][1])
        pack.add_entry(event)

        # print(f"{len(token_spans)} tokens in the article")
        for token_begin, token_end, role, text, token_text in collection["spans"]:
            char_begin = token_spans[token_begin][0]
            char_end = token_spans[token_end - 1][1]
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

    def _parse_pack(self, root: Any) -> Iterator[PackType]:
        # doc_name, ent_spans, evt_triggers, tokenized_text, gold_evt_links = collection
        pack = DataPack(root["doc_key"])

        doc_text, token_spans, sentence_spans = construct_text(root, "sentences")

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
