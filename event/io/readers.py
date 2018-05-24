import os
import logging
from collections import defaultdict
import torch
import numpy as np
import pickle
import random
import json
from collections import Counter
from event.arguments import consts
import torch.nn.functional as F


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
    def __init__(self, event_vocab, lookups, oovs, batch_size,
                 multi_context=False, max_events=200):
        """
        Reading the hashed dataset into cloze tasks.
        :param event_vocab: Event token vocabulary.
        :param batch_size: Number of cloze tasks per batch.
        :param multi_context: Whether to use multiple events as context.
        :param max_events: Number of events to keep per document.
        """
        self.batch_size = batch_size
        self.multi_context = multi_context
        self.max_events = max_events
        self.event_vocab = event_vocab
        self.lookups = lookups
        self.oovs = oovs

        self.event_padding = len(event_vocab)

        self.slot_names = ['subj', 'obj', 'prep', ]

        self.__data_types = {
            'gold_event': np.int64,
            'gold_distances': np.float32,
            'gold_features': np.float32,
            'cross_event': np.int64,
            'cross_distances': np.float32,
            'cross_features': np.float32,
            'inside_event': np.int64,
            'inside_distances': np.float32,
            'inside_features': np.float32,
        }

    def read_cloze_batch(self, data_in):
        clozes = defaultdict(list)

        max_context_size = 0

        cloze_count = 0
        for cloze_task in self.read_clozes(data_in):
            for key, value in cloze_task.items():
                clozes[key].append(value)

            if len(cloze_task['context']) > max_context_size:
                max_context_size = len(cloze_task['context'])

            cloze_count += 1

            if cloze_count == self.batch_size:
                cloze_count = 0
                instance_data = defaultdict(dict)
                for key, value in clozes.items():
                    if '_' in key:
                        inst_type, inst_key = key.split('_')
                        instance_data[inst_type][inst_key] = \
                            self._batch_combine(value)
                    elif key == 'context':
                        padded = [self._pad_context(v, max_context_size) for v
                                  in value]
                        context = self._batch_combine(padded)

                yield instance_data, context
                clozes = defaultdict(list)
                max_context_size = 0

    def _batch_combine(self, l_data):
        return torch.cat([torch.unsqueeze(d, 0) for d in l_data], dim=0)

    def _pad_context(self, context, pad_to_length):
        pad_len = pad_to_length - context.shape[0]
        padded = F.pad(context, (0, 0, 0, pad_len), 'constant', 0)
        return padded

    def _take_event_parts(self, event_info):
        event_components = [event_info['predicate'], event_info['frame']]

        for slot_name in self.slot_names:
            slot = event_info['slots'][slot_name]
            event_components.append(slot['fe'])
            event_components.append(slot['arg'])

        return [c if c > 0 else self.event_padding for c in event_components]

    def read_clozes(self, data_in):
        for line in data_in:
            doc_info = json.loads(line)
            features_by_eid = {}
            entity_heads = {}
            for eid, content in doc_info['entities'].items():
                features_by_eid[int(eid)] = content['features']
                entity_heads[int(eid)] = content['entity_head']

            # Organize all the arguments.
            event_data = []
            event_args = defaultdict(dict)
            entity_mentions = defaultdict(list)
            arg_entities = set()

            eid_count = Counter()

            for evm_index, event in enumerate(doc_info['events']):
                if evm_index == self.max_events:
                    break

                for slot, arg in event['args'].items():
                    # print(arg)
                    # Argument for nth event, at slot position 'slot'.
                    eid = arg['entity_id']
                    event_args[evm_index][slot] = (eid, arg['dep'])
                    # From eid to entity information.
                    entity_mentions[eid].append(
                        ((evm_index, slot), arg['sentence_id'])
                    )

                    if eid > 0:
                        arg_entities.add((evm_index, eid))
                        eid_count[eid] += 1

                event_data.append(
                    self.get_event_info(doc_info, evm_index,
                                        event_args[evm_index]))

            arg_entities = list(arg_entities)

            for evm_index, event in enumerate(doc_info['events']):
                if evm_index == self.max_events:
                    break

                for slot_index, slot in enumerate(self.slot_names):
                    arg = event['args'][slot]
                    correct_id = arg['entity_id']
                    # if arg['resolvable']:
                    # We re-count for resolvable, because we may filter events,
                    # which will change the resolvable attribute.
                    if eid_count[correct_id] > 1:
                        correct_id = arg['entity_id']
                        current_sent = arg['sentence_id']

                        gold_info = event_data[evm_index]

                        cross_instance, cross_filler_id = self.cross_cloze(
                            event_args, arg_entities, evm_index, slot,
                            correct_id)

                        inside_instance, inside_filler_id = self.inside_cloze(
                            event_args, evm_index, slot, correct_id)

                        cross_info = self.get_event_info(doc_info, evm_index,
                                                         cross_instance)

                        inside_info = self.get_event_info(doc_info, evm_index,
                                                          inside_instance)

                        cross_features = features_by_eid[cross_filler_id]
                        inside_features = features_by_eid[inside_filler_id]
                        gold_features = features_by_eid[correct_id]

                        origin_distances = self.compute_distances(
                            evm_index, entity_mentions, gold_info, current_sent,
                        )

                        cross_distances = self.compute_distances(
                            evm_index, entity_mentions, cross_info,
                            current_sent,
                        )

                        inside_distances = self.compute_distances(
                            evm_index, entity_mentions, inside_info,
                            current_sent,
                        )

                        # Events are represented as a list of "event words" now.
                        if not self.multi_context:
                            l_context = np.asarray(
                                [
                                    self._take_event_parts(
                                        self.sample_ignore_index(
                                            event_data, evm_index)
                                    )
                                ]
                            )

                        else:
                            l_context = [
                                np.asarray(
                                    self._take_event_parts(e)
                                ) for (i, e) in enumerate(event_data) if
                                not i == evm_index
                            ]

                        context_data = torch.from_numpy(np.stack(l_context))

                        cloze = {
                            'gold_event': self._take_event_parts(gold_info),
                            'gold_distances': origin_distances,
                            'gold_features': gold_features + [slot_index],
                            'cross_event': self._take_event_parts(cross_info),
                            'cross_distances': cross_distances,
                            'cross_features': cross_features + [slot_index],
                            'inside_event': self._take_event_parts(inside_info),
                            'inside_distances': inside_distances,
                            'inside_features': inside_features + [slot_index],
                        }

                        for key, value in cloze.items():
                            cloze[key] = self.to_torch(key, value)
                        cloze['context'] = context_data

                        yield cloze

    def to_torch(self, key, data):
        return torch.from_numpy(np.asarray(data, self.__data_types[key]))

    def sample_ignore_item(self, data, ignored_item):
        if len(data) <= 1:
            raise ValueError("Sampling from a list of size %d" % len(data))
        while True:
            sampled_item = random.choice(data)
            if not sampled_item == ignored_item:
                break
        return sampled_item

    def sample_ignore_index(self, data, ignored_index):
        if len(data) <= 1:
            raise ValueError("Sampling from a list of size %d" % len(data))
        while True:
            sampled_index = random.choice(range(len(data)))
            if not sampled_index == ignored_index:
                break
        return data[sampled_index]

    def get_event_info(self, doc_info, evm_index, arguments):
        event = doc_info['events'][evm_index]
        full_info = {
            'predicate': event['predicate'],
            'frame': event['frame'],
            'slots': {},
        }

        for slot, (eid, _) in arguments.items():
            if eid == -1:
                # This is an empty slot. Using
                pad = len(self.event_vocab)
                fe = pad
                arg_role = pad
            else:
                arg_info = event['args'][slot]
                fe = arg_info['fe']
                arg_role = arg_info['arg']

                if fe == -1:
                    fe = self.event_vocab[consts.unk_fe]

            full_info['slots'][slot] = {
                'arg': arg_role,
                'fe': fe,
                'context': event['args'][slot]['context'],
                'eid': eid
            }

        return full_info

    def compute_distances(self, current_evm_id, entity_sents, event,
                          sentence_id):
        """
        Compute the distance signature of the instance's other mentions to the
        sentence.
        :param current_evm_id:
        :param entity_sents:
        :param event:
        :param sentence_id:
        :return:
        """
        distances = []

        # Now use a large distance to represent Indefinite.
        # Infinity: if the entity cannot be found again, or it is not an entity.
        inf = 10000.0

        for current_slot, event_info in event['slots'].items():
            entity_id = event_info['eid']

            if entity_id == -1:
                # This is an empty slot.
                distances.append((inf, inf, inf))
                continue

            max_dist = 0.0
            min_dist = inf
            total_dist = 0.0
            total_pair = 0.0

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
                distances.append((inf, inf, inf))
                # print("No other mentions found")

        # Flatten the (argument x type) distances into a flat list.
        return [d for l in distances for d in l]

    def cross_cloze(self, event_args, arg_entities, current_evm, current_slot,
                    correct_id):
        """
        A negative cloze instance that use arguments from other events.
        :param event_args: List of all origin event arguments.
        :param arg_entities: Map from argument to entities.
        :param current_evm: The event id.
        :param current_slot: The slot id.
        :param correct_id: Correct entity id.
        :return:
        """
        wrong_evm, wrong_id = self.sample_ignore_item(
            arg_entities, (current_evm, correct_id)
        )
        neg_instance = {}
        neg_instance.update(event_args[current_evm])
        dep = event_args[current_evm][current_slot]
        neg_instance[current_slot] = (wrong_id, dep)

        return neg_instance, wrong_id

    def inside_cloze(self, event_args, current_evm, current_slot, correct_id):
        """
        A negative cloze instance that use arguments within the event.
        :param event_args:
        :param current_evm:
        :param current_slot:
        :param correct_id:
        :return:
        """
        current_event = event_args[current_evm]

        neg_instance = {}
        neg_instance.update(current_event)

        swapping_slot = self.sample_ignore_item(
            list(current_event.keys()), current_slot
        )
        swapping_id = current_event[swapping_slot]

        current_dep = event_args[current_evm][current_slot]
        swapping_dep = event_args[current_evm][swapping_slot]

        # Swap the two slots, this may create:
        # 1. instance with frames swapped
        # 2. instance with a frame moved to another empty slot
        neg_instance[swapping_slot] = (correct_id, swapping_dep)
        neg_instance[current_slot] = (swapping_id, current_dep)

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
                        'representEntityHead': ent['representEntityHead'],
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
