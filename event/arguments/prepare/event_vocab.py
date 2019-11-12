from collections import defaultdict, Counter
import os
import gzip
import json
import pickle
from json.decoder import JSONDecodeError
import logging
from event import util
from event.arguments.prepare.slot_processor import get_simple_dep

logger = logging.getLogger(__name__)


class TypedEventVocab:
    unk_predicate = 'unk_predicate-pred'
    unk_arg_word = 'unk_argument'
    unk_frame = 'unk_frame'
    unk_fe = 'unk_fe'
    unk_prep = 'unk_preposition'
    unk_dep = 'unk_dep'

    def __init__(self, vocab_dir, event_data=None):
        self.lookups = {}
        self.oovs = {}

        self.vocab_dir = vocab_dir

        if not os.path.exists(os.path.join(vocab_dir, 'predicate.vocab')):
            if event_data is None:
                logging.error("Vocabulary file not exist and not data "
                              "provided for counting.")

            logger.info("Counting vocabulary.")
            vocab_counters = self.get_vocab_count(event_data)
            for key, counter in vocab_counters.items():
                raw_vocab_path = os.path.join(vocab_dir, key + '.vocab')
                with open(raw_vocab_path, 'w') as out:
                    for key, value in counter.most_common():
                        out.write('{}\t{}\n'.format(key, value))
            logger.info("Done vocabulary counting.")

            # Now filter the vocabulary.
            logger.info("Filtering vocabulary.")
            self.filter_vocab(vocab_counters)
            logger.info("Done filtering.")
        else:
            logger.info("Will not overwrite vocabulary, using existing.")
            for f in os.listdir(vocab_dir):
                if '_' in f and f.endswith('.vocab'):
                    vocab_type = f.split('_')[0]
                elif f == 'frame.vocab':
                    vocab_type = 'frame'
                else:
                    continue

                self.lookups[vocab_type] = {}
                self.oovs[vocab_type] = 'unk_' + vocab_type

                with open(os.path.join(vocab_dir, f)) as vocab_file:
                    index = 0
                    for line in vocab_file:
                        word, count = line.strip().split('\t')
                        self.lookups[vocab_type][word] = index
                        index += 1

                logger.info(
                    "Loaded {} types for {}".format(
                        len(self.lookups[vocab_type]), vocab_type))

    def get_vocab_word(self, word, key):
        if not word:
            return self.oovs[key]

        if word in self.lookups[key]:
            return word
        else:
            return self.oovs[key]

    def make_arg(self, text, role):
        if role == 'NA':
            return text + "-" + self.unk_dep
        else:
            return text + "-" + role

    @staticmethod
    def make_predicate(text):
        return text.lower() + "-pred"

    @staticmethod
    def make_fe(frame, fe):
        return frame + ',' + fe

    def get_arg_entity_rep(self, arg, entity_text):
        rep = self.get_vocab_word(entity_text, 'argument')
        if rep == self.oovs['argument']:
            if 'ner' in arg:
                rep = arg['ner']

        # If still.
        if rep == self.oovs['argument']:
            # Fall back to use the argument's own text.
            rep = self.get_vocab_word(arg['text'], 'argument')

        if rep == self.oovs['argument']:
            if '-' in entity_text:
                rep = self.get_vocab_word(entity_text.split('-')[-1],
                                          'argument')

        if rep == self.oovs['argument']:
            if '-' in arg['text']:
                rep = self.get_vocab_word(arg['text'].split('-')[-1],
                                          'argument')

        return rep

    def get_unk_arg_rep(self):
        # TODO: This will create a full unknown argument, but we can back off to
        # a partial unknown argument in many cases.
        return self.make_arg(self.unk_arg_word, self.unk_dep)

    def get_arg_rep_no_dep(self, entity_rep):
        """Return the backoff version of the argument representation by using the
        unk_dep, but the actual entity.

        Args:
          entity_rep:

        Returns:
          

        """
        return self.make_arg(entity_rep, self.unk_dep)

    def get_arg_rep(self, arg_dep, entity_rep):
        if arg_dep.startswith('prep'):
            arg_dep = self.get_vocab_word(arg_dep, 'preposition')
        arg_rep = self.make_arg(entity_rep, arg_dep)
        return arg_rep

    # TODO: This will only use the verb form if the nominal form does not
    #  presented. Not sure which representation works better here.
    def get_pred_rep(self, event):
        pred = self.get_vocab_word(event['predicate'], 'predicate')

        if pred == self.oovs['predicate']:
            # Try to see if the verb form help.
            if 'verb_form' in event:
                pred = self.get_vocab_word(event['verb_form'], 'predicate')
        return self.make_predicate(pred)

    def get_fe_rep(self, frame_name, fe_role):
        # return self.make_fe(frame_name, fe_role)
        return self.get_vocab_word(self.make_fe(frame_name, fe_role), 'fe')

    @staticmethod
    def filter_by_count(counter, min_count):
        return [(key, count) for key, count in counter.most_common() if
                count >= min_count]

    def filter_vocab(self, vocab_counters,
                     top_num_prep=150,
                     min_token_count=500,
                     min_fe_count=50):
        filtered_vocab = {
            'predicate_min_%d' % min_token_count:
                self.filter_by_count(vocab_counters['predicate'],
                                     min_token_count),
            'argument_min_%d' % min_token_count:
                self.filter_by_count(vocab_counters['argument'],
                                     min_token_count),
            'preposition_top_%d' % top_num_prep:
                vocab_counters['preposition'].most_common(top_num_prep),
            'fe_min_%d' % min_fe_count:
                self.filter_by_count(vocab_counters['fe'], min_fe_count),
        }

        # Frames are retained if their frame elements are retained.
        fe_counter = filtered_vocab['fe_min_%d' % min_fe_count]
        frame_counter = Counter()
        for full_fe, count in fe_counter:
            frame, fe = full_fe.split(',')
            frame_counter[frame] += count
        filtered_vocab['frame'] = frame_counter.most_common()

        for key, counts in filtered_vocab.items():
            name = key.split('_')[0]

            oov = 'unk_' + name
            counts.insert(0, (oov, 0))

            self.lookups[name] = {}
            self.oovs[name] = oov

            index = 0
            for term, _ in counts:
                self.lookups[name][term] = index
                index += 1

        for key, vocab in filtered_vocab.items():
            with open(os.path.join(self.vocab_dir, key + '.vocab'), 'w') as out:
                for token, count in vocab:
                    out.write('{}\t{}\n'.format(token, count))

        with open(os.path.join(self.vocab_dir, 'lookups.pickle'), 'wb') as out:
            pickle.dump(self.lookups, out)

    def get_vocab_count(self, data_path):
        vocab_counters = defaultdict(Counter)

        doc_count = 0
        event_count = 0

        with gzip.open(data_path) as data:
            for line in data:
                doc_info = json.loads(line)

                represent_by_id = {}
                for entity in doc_info['entities']:
                    eid = entity['entityId']
                    represent = entity['representEntityHead']
                    represent_by_id[eid] = represent

                for event in doc_info['events']:
                    event_count += 1

                    predicate = event['predicate']
                    vocab_counters['predicate'][predicate] += 1

                    for arg in event['arguments']:
                        fe_name = arg['feName']
                        syn_role = arg['dep']
                        eid = arg['entityId']
                        arg_text = represent_by_id.get(eid, arg['text'])

                        vocab_counters['argument'][arg_text] += 1

                        if not fe_name == 'NA':
                            vocab_counters['fe'][
                                self.make_fe(event['frame'], fe_name)
                            ] += 1

                        if syn_role.startswith('prep'):
                            vocab_counters['preposition'][syn_role] += 1

                doc_count += 1
                if doc_count % 1000 == 0:
                    print('\rCounted vocab for {} events in '
                          '{} docs.'.format(event_count, doc_count), end='')

        return vocab_counters


class EmbbedingVocab:
    def __init__(self, vocab_file, with_padding=False, extras=None):
        self.vocab_file = vocab_file
        self.vocab = {}
        self.tf = []
        self.extras = []
        self.pad = '__PADDING__'

        if extras:
            for name in extras:
                self.add_extra(name)

        self.add_extra(self.pad)

        self.__read_vocab()

    def get_index(self, token, unk):
        try:
            return self.vocab[token]
        except KeyError:
            if unk:
                return self.vocab[unk]
            else:
                return -1

    def extra_size(self):
        return len(self.extras)

    def add_extra(self, name):
        """Add extra dimensions into the embedding vocab, used for special
        tokens.

        Args:
          name:

        Returns:

        """
        if name in self.extras:
            raise ValueError(
                "Cannot add extras to an embedding with the same name")
        self.extras.append(name)
        extra_index = len(self.vocab)
        self.vocab[name] = extra_index
        self.tf.append(0)

        logger.info(f"Adding {name} as extra dimension {extra_index}")

        return extra_index

    def get_size(self):
        return len(self.vocab)

    def vocab_items(self):
        return self.vocab.items()

    def get_term_freq(self, token):
        return self.tf[self.get_index(token, None)]

    def __read_vocab(self):
        with open(self.vocab_file) as din:
            index = len(self.vocab)
            for line in din:
                word, count = line.split()
                self.vocab[word] = index
                self.tf.append(int(count))
                index += 1


def create_sentences(doc, event_vocab, output_path, include_frame=False,
                     simple_dep=False):
    if include_frame:
        print("Adding frames to sentences.")

    doc_count = 0
    event_count = 0

    with gzip.open(doc) as data, gzip.open(output_path, 'w') as out:
        for line in data:
            try:
                doc_info = json.loads(line)
            except JSONDecodeError:
                continue

            sentence = []

            represent_by_id = {}
            for entity in doc_info['entities']:
                eid = entity['entityId']
                represent = entity['representEntityHead']
                represent_by_id[eid] = represent

            for event in doc_info['events']:
                event_count += 1

                sentence.append(event_vocab.get_pred_rep(event))

                if include_frame:
                    frame = event_vocab.get_vocab_word(event['frame'], 'frame')
                    sentence.append(frame)

                for arg in event['arguments']:
                    dep = arg['dep']

                    if simple_dep:
                        dep = get_simple_dep(dep)

                    if (not include_frame) and dep == 'NA':
                        continue

                    sentence.append(
                        event_vocab.get_arg_rep(
                            dep,
                            event_vocab.get_arg_entity_rep(
                                arg, represent_by_id[arg['entityId']]
                            )
                        )
                    )

                    if include_frame:
                        sentence.append(
                            event_vocab.get_fe_rep(frame, arg['feName'])
                        )

            doc_count += 1

            out.write(str.encode(' '.join(sentence) + '\n'))

            if event_count % 1000 == 0:
                print('\rCreated sentences for {} documents, '
                      '{} events.'.format(doc_count, event_count), end='')

    print('\rCreated sentences for {} documents, '
          '{} events.\n'.format(doc_count, event_count), end='')


def main(event_data, vocab_dir, sent_out):
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    event_vocab = TypedEventVocab(vocab_dir, event_data=event_data)
    logger.info("Done loading vocabulary.")

    logger.info("Creating event sentences")
    sent_out_simple = sent_out + '_simple'

    if not os.path.exists(sent_out):
        os.makedirs(sent_out)

    if not os.path.exists(sent_out_simple):
        os.makedirs(sent_out_simple)

    sent_out_pred_only = os.path.join(sent_out, 'sent_pred_only.gz')
    if not os.path.exists(sent_out_pred_only):
        create_sentences(event_data, event_vocab, sent_out_pred_only,
                         include_frame=False)
    else:
        logger.info(f"Will not overwrite {sent_out_pred_only}")

    sent_out_with_frame = os.path.join(sent_out, 'sent_with_frames.gz')
    if not os.path.exists(sent_out_with_frame):
        create_sentences(event_data, event_vocab, sent_out_with_frame,
                         include_frame=True)
    else:
        logger.info(f"Will not overwrite {sent_out_with_frame}")

    sent_out_pred_only_simple = os.path.join(sent_out_simple,
                                             'sent_pred_only.gz')
    if not os.path.exists(sent_out_pred_only_simple):
        create_sentences(event_data, event_vocab, sent_out_pred_only_simple,
                         include_frame=False, simple_dep=True)
    else:
        logger.info(f"Will not overwrite {sent_out_pred_only_simple}")

    sent_out_with_frame_simple = os.path.join(sent_out_simple,
                                              'sent_with_frames.gz')
    if not os.path.exists(sent_out_with_frame_simple):
        create_sentences(event_data, event_vocab, sent_out_with_frame_simple,
                         include_frame=True, simple_dep=True)
    else:
        logger.info(f"Will not overwrite {sent_out_with_frame_simple}")


if __name__ == '__main__':
    parser = util.OptionPerLineParser(description='Event Vocabulary.',
                                      fromfile_prefix_chars='@')
    parser.add_argument('--vocab_dir', type=str, help='Vocabulary directory.')
    parser.add_argument('--input_data', type=str, help='Input data.')
    parser.add_argument('--sent_out', type=str, help='Sentence out dir.')

    util.set_basic_log()

    args = parser.parse_args()
    main(
        args.input_data, args.vocab_dir, args.sent_out
    )
