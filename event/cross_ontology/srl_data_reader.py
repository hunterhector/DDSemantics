import os
import numpy as np
import json
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
from collections import defaultdict


class SrlDataReader:
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = TreebankWordTokenizer()

    def read_data(self, data_dir):
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                if not name.endswith(".json"):
                    continue

                full_path = os.path.join(root, name)

                with open(full_path) as fin:
                    doc = json.load(fin)
                    for data in self.parse_doc(doc):
                        yield data

    def parse_doc(self, doc):
        text = doc["text"]
        events = doc["events"]
        fillers = doc["fillers"]
        entities = doc["entities"]

        begin_map = defaultdict(list)
        end_map = defaultdict(list)

        for f in fillers:
            begin_map[f["begin"]].append((f["id"], f["type"]))
            end_map[f["end"]].append((f["id"], f["type"]))

        for entity in entities:
            for em in entity["mentions"]:
                begin_map[em["begin"]].append((em["id"], em["type"]))
                end_map[em["end"]].append((em["id"], em["type"]))

        for event in events:
            for evm in event["mentions"]:
                begin_map[evm["begin"]].append((evm["id"], evm["type"]))
                end_map[evm["end"]].append((evm["id"], evm["type"]))

        indexed_doc = []
        tags = []
        sent_offset = 0

        on_types = set()

        for sentence in text.split("\n"):
            word_spans = self.tokenizer.span_tokenize(sentence)

            tokens = []

            for b, e in word_spans:
                token_text = sentence[b:e]
                indexed_doc.append(self.vocab.get(token_text, 0))

                begin = sent_offset + b
                end = sent_offset + e

                token_tags = []

                for begin_obj in begin_map[begin]:
                    obj_id, obj_type = begin_obj
                    token_tags.append("B_" + obj_type)
                    on_types.add(obj_type)

                for end_obj in end_map[end]:
                    obj_id, obj_type = end_obj
                    token_tags.append("I_" + obj_type)
                    on_types.remove(obj_type)

                if on_types:
                    for t in on_types:
                        token_tags.append("I_" + t)
                else:
                    token_tags.append("O")

                tags.append(token_tags)
                tokens.append(self.vocab.get(token_text))

            sent_offset += len(sentence) + 1

            print(tokens)
            input(tags)

            yield tokens, tags
