import os
import logging
from collections import defaultdict
import torch
import numpy as np
import pickle
import random
import json
from collections import Counter
import sys


class Vocab:
    def __init__(self, base_folder, name, embedding_path=None, emb_dim=100):
        self.fixed = False
        self.base_folder = base_folder
        self.name = name

        if self.load_map():
            logging.info("Loaded existing vocabulary mapping.")
            self.fix()
        else:
            logging.info("Creating new vocabulary mapping file.")
            self.token2i = defaultdict(lambda: len(self.token2i))

        self.unk = self.token2i["<unk>"]

        if embedding_path:
            logging.info("Loading embeddings from %s." % embedding_path)
            self.embedding = self.load_embedding(embedding_path, emb_dim)
            self.fix()

        self.i2token = dict([(v, k) for k, v in self.token2i.items()])

    def __call__(self, *args, **kwargs):
        return self.token_dict()[args[0]]

    def load_embedding(self, embedding_path, emb_dim):
        with open(embedding_path, 'r') as f:
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
        path = os.path.join(self.base_folder, self.name + '.pickle')
        if not os.path.exists(path):
            with open(path, 'wb') as p:
                pickle.dump(dict(self.token2i), p)

    def load_map(self):
        path = os.path.join(self.base_folder, self.name + '.pickle')
        if os.path.exists(path):
            with open(path, 'rb') as p:
                self.token2i = pickle.load(p)
                return True
        else:
            return False


class ConllUReader:
    def __init__(self, data_files, config, token_vocab, tag_vocab, language):
        self.experiment_folder = config.experiment_folder
        self.data_files = data_files
        self.data_format = config.format

        self.no_punct = config.no_punct
        self.no_sentence = config.no_sentence

        self.batch_size = config.batch_size

        self.window_sizes = config.window_sizes
        self.context_size = config.context_size

        logging.info("Batch size is %d, context size is %d." % (
            self.batch_size, self.context_size))

        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab

        self.language = language

        logging.info("Corpus with [%d] words and [%d] tags.",
                     self.token_vocab.vocab_size(),
                     self.tag_vocab.vocab_size())

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
                parsed_data = (
                    token_ids, tag_ids, features, token_meta
                )

                sent_start = (-1, -1)
                sent_end = (-1, -1)

                for line in data:
                    if line.startswith("#"):
                        if line.startswith("# newdoc"):
                            docid = line.split("=")[1].strip()
                    elif not line.strip():
                        # Yield data when seeing sentence break.
                        yield parsed_data, (
                            sentence_id, (sent_start[1], sent_end[1]), docid
                        )
                        [d.clear() for d in parsed_data]
                        sentence_id += 1
                    else:
                        parts = line.split()
                        _, token, lemma, _, pos, _, head, dep, _, tag \
                            = parts[:10]
                        lemma = lemma.lower()
                        pos = pos.lower()

                        span = [int(x) for x in parts[-1].split(",")]

                        if pos == 'punct' and self.no_punct:
                            # Simulate the non-punctuation audio input.
                            continue

                        parsed_data[0].append(self.token_vocab(token.lower()))
                        parsed_data[1].append(self.tag_vocab(tag))
                        parsed_data[2].append(
                            (lemma, pos, head, dep)
                        )
                        parsed_data[3].append(
                            (token, span)
                        )

                        if not sentence_id == sent_start[0]:
                            sent_start = [sentence_id, span[0]]

                        sent_end = [sentence_id, span[1]]

    def read_window(self):
        for (token_ids, tag_ids, features, token_meta), meta in self.parse():
            assert len(token_ids) == len(tag_ids)

            token_pad = [self.token_vocab.unk] * self.context_size
            tag_pad = [self.tag_vocab.unk] * self.context_size

            feature_pad = ["EMPTY"] * self.context_size

            actual_len = len(token_ids)

            token_ids = token_pad + token_ids + token_pad
            tag_ids = tag_pad + tag_ids + tag_pad
            features = feature_pad + features + feature_pad
            token_meta = feature_pad + token_meta + feature_pad

            for i in range(actual_len):
                start = i
                end = i + self.context_size * 2 + 1
                yield token_ids[start: end], tag_ids[start:end], \
                      features[start:end], token_meta[start:end], meta

    def convert_batch(self):
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


class HashedClozeReader:
    def __init__(self):
        pass

    def read_clozes(self, data_in):
        for line in data_in:
            doc_info = json.loads(line)
            features_by_eid = {}
            for eid, content in doc_info['entities'].items():
                features_by_eid[eid] = content['features']

            # Organize all the arguments.
            event_args = defaultdict(dict)
            entity_mentions = defaultdict(list)
            arg_entities = set()
            for evm_index, event in enumerate(doc_info['events']):
                for slot, arg in event['args'].items():
                    # Argument for nth event, at slot position 'slot'.
                    eid = arg['entity_id']
                    event_args[evm_index][slot] = eid
                    # The sentence position of the entities.
                    entity_mentions[eid].append(
                        ((evm_index, slot), arg['sentence_id'])
                    )

                    if eid > 0:
                        arg_entities.add((evm_index, eid))

            for evm_index, event in enumerate(doc_info['events']):
                for slot, arg in event['args'].items():
                    if arg['resolvable']:
                        correct_id = arg['entity_id']
                        current_sent = arg['sentence_id']
                        correct_instance = event_args[evm_index]

                        cross_instance, cross_filler_id = self.cross_cloze(
                            event_args, arg_entities, evm_index, slot,
                            correct_id)

                        inside_instance, inside_filler_id = self.inside_cloze(
                            event_args, evm_index, slot, correct_id)

                        gold_info = self.get_instance_info(
                            doc_info, correct_id, evm_index,
                            entity_mentions, correct_instance, current_sent,
                        )

                        cross_info = self.get_instance_info(
                            doc_info, cross_filler_id, evm_index,
                            entity_mentions, cross_instance, current_sent,
                        )

                        inside_info = self.get_instance_info(
                            doc_info, inside_filler_id, evm_index,
                            entity_mentions, inside_instance, current_sent,
                        )

                        yield gold_info, cross_info, inside_info

    def get_instance_info(self, doc_info, filler_id, evm_index,
                          entity_mentions, instance, sent):
        full_info = {
            'features': doc_info['entities'][str(filler_id)]['features'],
            'slots': {},
        }

        origin_distances = self.compute_distances(
            evm_index, entity_mentions, instance, sent,
        )

        for (slot, eid), dist in zip(instance.items(), origin_distances):
            fe = doc_info['events'][evm_index]['args'][slot]['fe']
            full_info['slots'][slot] = {
                'fe': fe,
                'distance': dist,
            }
        return full_info

    def compute_distances(self, current_evm_id, entity_sents, instance,
                          sentence_id):
        """
        Compute the distance signature of the instance's other mentions to the
        sentence.
        :param current_evm_id:
        :param entity_sents:
        :param instance:
        :param sentence_id:
        :return:
        """
        # print("Computing distance for ", instance, 'sentence ', sentence_id)

        distances = []

        for current_slot, entity_id in instance.items():
            if entity_id == -1:
                # This is an empty slot.
                distances.append((-1, -1, -1))
                continue

            max_dist = 0
            min_dist = float('inf')
            total_dist = 0
            total_pair = 0

            for (evm_id, slot), sid in entity_sents[entity_id]:
                if evm_id == current_evm_id:
                    # Do not compute mentions at the same event.
                    continue

                distance = abs(sid - sentence_id)
                if distance < min_dist:
                    min_dist = distance
                if distance > max_dist:
                    max_dist = distance
                total_dist += distance
                total_pair += 1.0

            if total_pair > 0:
                distances.append((max_dist, min_dist, total_dist / total_pair))
                # print(max_dist, min_dist, total_dist / total_pair)
            else:
                # This argument is not seen elsewhere, it should be a special
                # distance label.
                distances.append((float('inf'), float('inf'), float('inf')))
                # print("No other mentions found")

        return distances

    def cross_cloze(self, event_args, arg_entities, current_evm, current_pos,
                    correct_id):
        """
        A negative cloze instance that use arguments from other events.
        :param event_args:
        :param arg_entities:
        :param current_evm:
        :param current_pos:
        :param correct_id:
        :return:
        """
        candidates = []

        for evm, ent in arg_entities:
            if not correct_id == ent and not current_evm == evm:
                candidates.append(ent)

        wrong_id = random.choice(candidates)

        neg_instance = {}
        neg_instance.update(event_args[current_evm])
        neg_instance[current_pos] = wrong_id

        return neg_instance, wrong_id

    def inside_cloze(self, event_args, current_index, current_pos, correct_id):
        """
        A negative cloze instance that use arguments within the event.
        :param event_args:
        :param current_index:
        :param current_pos:
        :param correct_id:
        :return:
        """
        current_event = event_args[current_index]

        neg_instance = {}
        neg_instance.update(current_event)

        slots = []
        for slot, eid in current_event.items():
            # Exclude correct slot.
            if not slot == current_pos:
                slots.append(slot)

        # Select another slot.
        wrong_slot = random.choice(slots)
        wrong_id = current_event[wrong_slot]

        # Swap the two slots, this may create:
        # 1. instance with frames swapped
        # 2. instance with a frame moved to another empty slot
        neg_instance[wrong_slot] = correct_id
        neg_instance[current_pos] = wrong_id

        return neg_instance, correct_id


class EventReader:
    def __init__(self):
        self.target_roles = ['arg0', 'arg1', 'prep']
        self.entity_info_fields = ['syntactic_role', 'mention_text',
                                   'entity_id']
        self.entity_equal_fields = ['entity_id', 'represent']

        self.len_arg_fields = 4

    def get_context(self, sentence, start, end, window_size=5):
        right_tokens = sentence[end:].strip().split()
        right_win = min(window_size, len(right_tokens))
        right_context = right_tokens[:right_win]

        left_tokens = sentence[:start].strip().split()
        left_tokens.reverse()
        left_win = min(window_size, len(left_tokens))

        left_context = left_tokens[:left_win]
        left_context.reverse()

        return left_context, right_context

    def read_events(self, data_in):
        for line in data_in:
            doc = json.loads(line)
            docid = doc['docid']

            events = []

            eid_count = Counter()

            entity_heads = {}

            entities = {}

            if 'entities' in doc:
                for ent in doc['entities']:
                    entity_heads[ent['entityId']] = ent['representEntityHead']

                    entities[ent['entityId']] = {
                        'features': ent['entityFeatures'],
                    }

            for event_info in doc['events']:
                sent = doc['sentences'][event_info['sentenceId']]

                raw_context = self.get_context(
                    sent,
                    event_info['predicateStart'],
                    event_info['predicateEnd'],
                )

                event = {
                    'predicate': event_info['predicate'],
                    'predicate_context': raw_context,
                    # 'predicate_context': event_info['context'],
                    'frame': event_info.get('frame', 'NA'),
                    'arguments': [],
                    'predicate_start': event_info['predicateStart'],
                    'predicate_end': event_info['predicateEnd'],
                }

                events.append(event)

                for arg_info in event_info['arguments']:
                    if 'argStart' in arg_info:
                        arg_context = self.get_context(
                            sent, arg_info['argStart'], arg_info['argEnd']
                        )
                    else:
                        left, right = arg_info['context'].split('___')
                        arg_context = left.split(), right.split()

                    if entity_heads:
                        represent = entity_heads[arg_info['entityId']]
                    else:
                        represent = arg_info['representText']

                    arg = {
                        'dep': arg_info['dep'],
                        'fe': arg_info['feName'],
                        'arg_context': arg_context,
                        'represent': represent,
                        'entity_id': arg_info['entityId'],
                        'resolvable': False,
                        'arg_start': arg_info['argStart'],
                        'arg_end': arg_info['argEnd'],
                        'sentence_id': event_info['sentenceId']
                    }

                    eid_count[arg_info['entityId']] += 1
                    event['arguments'].append(arg)

            for event in events:
                for arg in event['arguments']:
                    if eid_count[arg['entity_id']] > 1:
                        arg['resolvable'] = True

            yield docid, events, entities

    def _same_entity(self, ent1, ent2):
        return any([ent1[f] == ent2[f] for f in self.entity_equal_fields])

    def _entity_info(self, arg):
        return dict([(k, arg[k]) for k in self.entity_info_fields])
