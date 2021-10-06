import os
import logging
from collections import defaultdict, Counter
import numpy as np
import pickle
import json


def revert_nmod(dep):
    if dep.startswith("nmod:") and not dep == "nmod:'s":
        dep = dep.replace("nmod:", "prep_")
    return dep


class Vocab:
    def __init__(
        self, base_folder, name, embedding_path=None, emb_dim=100, ignore_existing=False
    ):
        self.fixed = False
        self.base_folder = base_folder
        self.name = name

        if not ignore_existing and self.load_map():
            self.fix()
        else:
            logging.info("Creating new vocabulary mapping file.")
            self.token2i = defaultdict(lambda: len(self.token2i))

        self.unk = self.token2i["<unk>"]
        self.pad = self.token2i["<padding>"]

        if embedding_path:
            logging.info("Loading embeddings from %s." % embedding_path)
            self.embedding = self.load_embedding(embedding_path, emb_dim)
            self.fix()

        self.i2token = dict([(v, k) for k, v in self.token2i.items()])

    def __call__(self, *args, **kwargs):
        token = args[0]
        index = self.token_dict()[token]
        self.i2token[index] = token
        return index

    def load_embedding(self, embedding_path, emb_dim):
        with open(embedding_path, "r") as f:
            emb_list = []
            for line in f:
                parts = line.split()
                word = parts[0]
                if len(parts) > 1:
                    embedding = np.array([float(val) for val in parts[1:]])
                else:
                    embedding = np.random.rand(1, emb_dim)

                self.token2i[word]
                emb_list.append(embedding)
            logging.info("Loaded %d words." % len(emb_list))
            return np.vstack(emb_list)

    def fix(self):
        # After fixed, the vocabulary won't grow.
        self.token2i = defaultdict(lambda: self.unk, self.token2i)
        self.fixed = True
        self.dump_map()

    def reveal_origin(self, token_ids):
        return [self.i2token[t] for t in token_ids]

    def token_dict(self):
        return self.token2i

    def vocab_size(self):
        return len(self.i2token)

    def dump_map(self):
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

        path = os.path.join(self.base_folder, self.name + ".pickle")
        if not os.path.exists(path):
            with open(path, "wb") as p:
                pickle.dump(dict(self.token2i), p)

    def load_map(self):
        path = os.path.join(self.base_folder, self.name + ".pickle")
        if os.path.exists(path):
            with open(path, "rb") as p:
                self.token2i = pickle.load(p)
                logging.info("Loaded existing vocabulary mapping at: {}".format(path))
                return True
        return False


class EventReader:
    def __init__(self):
        self.target_roles = ["arg0", "arg1", "prep"]
        self.entity_info_fields = ["syntactic_role", "mention_text", "entity_id"]
        self.entity_equal_fields = ["entity_id", "represent"]

        self.len_arg_fields = 4

    @staticmethod
    def get_context(text, start, end, window_size=5):
        right_tokens = text[end:].strip().split()
        right_win = min(window_size, len(right_tokens))
        right_context = right_tokens[:right_win]

        left_tokens = text[:start].strip().split()
        left_tokens.reverse()
        left_win = min(window_size, len(left_tokens))

        left_context = left_tokens[:left_win]
        left_context.reverse()

        return left_context, right_context

    def read_events(self, data_in, gold_role_field):
        JOINED_TEXT = "joined_text_format"
        SPLIT_SENT = "split_sent_format"

        for line in data_in:
            doc = json.loads(line)

            doc_text = ""
            sentence_spans = []

            if "text" in doc:
                doc_format = JOINED_TEXT
                doc_text = doc["text"]
                sentence_spans = doc["sentences"]
            elif "sentences" in doc:
                doc_format = SPLIT_SENT
                sentences = doc["sentences"]
                doc_text += "\n".join(sentences)

                offset = 0
                for sent in sentences:
                    b = offset
                    offset += len(sent)
                    e = offset
                    sentence_spans.append({"begin": b, "end": e})
                    offset += 1
            else:
                raise KeyError("Unknown document format.")

            docid = doc["docid"]

            events = []

            entity_spans = defaultdict(set)

            entity_heads = {}

            entities = {}

            if "entities" in doc:
                for ent in doc["entities"]:
                    entity_heads[ent["entityId"]] = ent["representEntityHead"]

                    entities[ent["entityId"]] = {
                        "features": ent["entityFeatures"],
                        "represent_entity_head": ent["representEntityHead"],
                    }

                    if "entityType" in ent:
                        entities[ent["entityId"]]["entity_type"] = ent["entityType"]

            for event_info in doc["events"]:
                pred_start, pred_end = (
                    event_info["predicateStart"],
                    event_info["predicateEnd"],
                )

                if doc_format == SPLIT_SENT:
                    arg_sent_id = event_info["sentenceId"]
                    pred_sent_begin = sentence_spans[arg_sent_id]["begin"]
                    pred_start += pred_sent_begin
                    pred_end += pred_sent_begin

                raw_context = self.get_context(doc_text, pred_start, pred_end)

                event = {
                    "predicate": event_info["predicate"],
                    "predicate_context": raw_context,
                    "frame": event_info.get("frame", "NA"),
                    "arguments": [],
                    "predicate_start": pred_start,
                    "predicate_end": pred_end,
                    "sentence_id": event_info["sentenceId"],
                    "event_type": event_info.get("eventType", "NA"),
                    "is_target": event_info.get("fromGC", False),
                }

                if "verbForm" in event_info:
                    event["verb_form"] = event_info["verbForm"]

                events.append(event)

                for arg_info in event_info["arguments"]:
                    left, right = arg_info["context"].split("___", 1)
                    arg_context = left.split(), right.split()

                    if entity_heads:
                        represent = entity_heads[arg_info["entityId"]]
                    else:
                        represent = arg_info["text"]

                    arg_start, arg_end = arg_info["argStart"], arg_info["argEnd"]

                    if doc_format == SPLIT_SENT:
                        # The deprecated split sentence format forgot the
                        # sentence id for arguments, we can still use the
                        # predicate id at training time.
                        arg_sent_id = event_info["sentenceId"]
                        arg_sent_begin = sentence_spans[arg_sent_id]["begin"]
                        arg_start += arg_sent_begin
                        arg_end += arg_sent_begin
                    elif doc_format == JOINED_TEXT:
                        arg_sent_id = arg_info["sentenceId"]

                    arg = {
                        "dep": revert_nmod(arg_info.get("dep", "NA")),
                        "fe": arg_info["feName"],
                        "arg_context": arg_context,
                        "entity_id": arg_info["entityId"],
                        "resolvable": False,
                        "arg_start": arg_start,
                        "arg_end": arg_end,
                        "sentence_id": arg_sent_id,
                        "role": arg_info.get("argument_role", "NA"),
                        "arg_phrase": arg_info["argumentPhrase"],
                        "text": arg_info["text"],
                        "represent": represent,
                        "source": arg_info.get("source", "automatic"),
                    }

                    gold_role = arg_info.get(gold_role_field, "NA")
                    if not gold_role == "NA":
                        arg["gold_role"] = gold_role

                    if "isImplicit" in arg_info:
                        arg["implicit"] = arg_info["isImplicit"]
                    if "isSucceeding" in arg_info:
                        arg["succeeding"] = arg_info["isSucceeding"]
                    if "isIncorporated" in arg_info:
                        arg["incorporated"] = arg_info["isIncorporated"]

                    if "ner" in arg_info:
                        arg["ner"] = arg_info["ner"]
                    else:
                        its_entity = entities[arg_info["entityId"]]
                        if "entity_type" in its_entity:
                            arg["ner"] = its_entity["entity_type"]

                    entity_spans[arg_info["entityId"]].add((arg_start, arg_end))

                    event["arguments"].append(arg)

            for event in events:
                for arg in event["arguments"]:
                    if len(entity_spans[arg["entity_id"]]) > 1:
                        arg["resolvable"] = True

            yield docid, events, entities, sentence_spans

    def _same_entity(self, ent1, ent2):
        return any([ent1[f] == ent2[f] for f in self.entity_equal_fields])

    def _entity_info(self, arg):
        return dict([(k, arg[k]) for k in self.entity_info_fields])


class ConllUReader:
    def __init__(
        self, data_files, config, token_vocab, tag_vocab, language, tag_index=-1
    ):
        self.data_files = data_files
        self.data_format = config.input_format

        self.no_punct = config.no_punct
        self.no_sentence = config.no_sentence

        self.batch_size = config.batch_size

        self.window_sizes = config.window_sizes
        self.context_size = config.context_size

        logging.info(
            "Batch size is %d, context size is %d."
            % (self.batch_size, self.context_size)
        )

        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab

        self.tag_index = tag_index

        self.language = language

        logging.info(
            "Corpus with [%d] words and [%d] tags.",
            self.token_vocab.vocab_size(),
            self.tag_vocab.vocab_size(),
        )

        self.__batch_data = []

    def parse(self):
        for data_file in self.data_files:
            logging.info("Loading data from [%s] " % data_file)
            with open(data_file) as data:
                sentence_id = 0

                token_ids = []
                tag_ids = []
                features = []
                token_meta = []
                parsed_data = (token_ids, tag_ids, features, token_meta)

                sent_start = (-1, -1)
                sent_end = (-1, -1)

                for line in data:
                    line = line.strip()
                    if line.startswith("#"):
                        if line.startswith("# newdoc"):
                            parts = line.split("=")
                            if len(parts) > 1:
                                docid = line.split("=")[1].strip()
                            else:
                                docid = os.path.basename(data_file)
                    elif not line:
                        # Yield data when seeing sentence break.
                        yield parsed_data, (
                            sentence_id,
                            (sent_start[1], sent_end[1]),
                            docid,
                        )
                        [d.clear() for d in parsed_data]
                        sentence_id += 1
                    else:
                        parts = line.split()

                        token = parts[1]
                        upos = parts[3]
                        xpos = parts[4]

                        # (wid, token, lemma, upos, xpos, feats, head, deprel,
                        #  deps) = parts[:9]

                        tag = (
                            parts[self.tag_index]
                            if self.tag_index >= 0
                            else self.tag_vocab.unk
                        )

                        span = [int(x) for x in parts[-1].split(",")]

                        if xpos == "PUNCT" or upos == "PUNCT":
                            if self.no_punct:
                                # Simulate the non-punctuation audio input.
                                continue

                        parsed_data[0].append(self.token_vocab(token.lower()))
                        parsed_data[1].append(self.tag_vocab(tag))

                        word_feature = parts[2:-1]

                        parsed_data[2].append(word_feature)

                        parsed_data[3].append((token, span))

                        if not sentence_id == sent_start[0]:
                            sent_start = [sentence_id, span[0]]

                        sent_end = [sentence_id, span[1]]

    def read_window(self):
        # empty_feature = ["EMPTY"] * self.feature_vector_len

        for (token_ids, tag_ids, features, token_meta), meta in self.parse():
            assert len(token_ids) == len(tag_ids)

            token_pad = [self.token_vocab.pad] * self.context_size
            tag_pad = [self.tag_vocab.pad] * self.context_size

            feature_pad = [None] * self.context_size

            actual_len = len(token_ids)

            token_ids = token_pad + token_ids + token_pad
            tag_ids = tag_pad + tag_ids + tag_pad
            features = feature_pad + features + feature_pad
            token_meta = feature_pad + token_meta + feature_pad

            for i in range(actual_len):
                start = i
                end = i + self.context_size * 2 + 1
                yield (
                    token_ids[start:end],
                    tag_ids[start:end],
                    features[start:end],
                    token_meta[start:end],
                    meta,
                )

    def convert_batch(self):
        import torch

        tokens, tags, features = zip(*self.__batch_data)
        tokens, tags = torch.FloatTensor(tokens), torch.FloatTensor(tags)
        if torch.cuda.is_available():
            tokens.cuda()
            tags.cuda()
        return tokens, tags

    def read_batch(self):
        for token_ids, tag_ids, features, meta in self.read_window():
            if len(self.__batch_data) < self.batch_size:
                self.__batch_data.append((token_ids, tag_ids, features))
            else:
                data = self.convert_batch()
                self.__batch_data.clear()
                return data

    def num_classes(self):
        return self.tag_vocab.vocab_size()
