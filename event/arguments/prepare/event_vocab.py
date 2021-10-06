from collections import defaultdict, Counter
import os
import gzip
import json
import pickle
from json.decoder import JSONDecodeError
import logging
from typing import Dict
import pdb

from event import util
from event.arguments.prepare.slot_processor import get_simple_dep, is_propbank_dep

logger = logging.getLogger(__name__)


class TypedEventVocab:
    unk_predicate = "unk_predicate-pred"
    unk_arg_word = "unk_argument"
    unk_frame = "unk_frame"
    unk_fe = "unk_fe"
    unk_prep = "unk_preposition"
    unk_dep = "unk_dep"
    unobserved_fe = "__unobserved_fe__"
    unobserved_arg = "__unobserved_arg__"
    ghost = "__ghost_component__"

    def __init__(self, vocab_dir, event_data=None):
        self.lookups: Dict[str, Dict[str, int]] = {}
        self.oovs: Dict[str, str] = {}

        self.vocab_dir = vocab_dir

        if not os.path.exists(os.path.join(vocab_dir, "predicate.vocab")):
            if event_data is None:
                logging.error(
                    "Vocabulary file not exist and not data " "provided for counting."
                )

            logger.info("Counting vocabulary.")
            vocab_counters = self.get_vocab_count(event_data)
            for vocab_name, counter in vocab_counters.items():
                raw_vocab_path = os.path.join(vocab_dir, vocab_name + ".vocab")
                with open(raw_vocab_path, "w") as out:
                    for key, value in counter.most_common():
                        out.write("{}\t{}\n".format(key, value))
            logger.info("Done vocabulary counting.")

            # Now filter the vocabulary.
            logger.info("Filtering vocabulary.")
            filtered_vocab = self.filter_vocab(vocab_counters)
            logger.info("Done filtering.")

            logger.info("Writing filtered vocab to disk.")
            for key, vocab in filtered_vocab.items():
                with open(os.path.join(self.vocab_dir, key + ".vocab"), "w") as out:
                    for token, count in vocab:
                        out.write("{}\t{}\n".format(token, count))

            self.pickle_counts()

            logger.info("Done.")
        else:
            logger.info("Will not overwrite vocabulary, using existing.")

            if not self.unpickle_counts():
                logger.info("Reading counts from .vocab files.")

                f_name: str
                for f_name in os.listdir(vocab_dir):
                    if "_" in f_name and f_name.endswith(".vocab"):
                        vocab_type = f_name.split("_")[0]
                    else:
                        continue

                    self.lookups[vocab_type] = {}
                    self.oovs[vocab_type] = "unk_" + vocab_type

                    with open(os.path.join(vocab_dir, f_name)) as vocab_file:
                        index = 0
                        for line in vocab_file:
                            word, count = line.strip().split("\t")
                            self.lookups[vocab_type][word] = index
                            index += 1

                    logger.info(
                        "Loaded {} types for {}".format(
                            len(self.lookups[vocab_type]), vocab_type
                        )
                    )

                self.pickle_counts()

    def pickle_counts(self):
        with open(os.path.join(self.vocab_dir, "lookups.pickle"), "wb") as out:
            pickle.dump(self.lookups, out)

        with open(os.path.join(self.vocab_dir, "oovs.pickle"), "wb") as out:
            pickle.dump(self.oovs, out)

    def unpickle_counts(self):
        lookup_pickle = os.path.join(self.vocab_dir, "lookups.pickle")
        oov_pickle = os.path.join(self.vocab_dir, "oovs.pickle")

        if os.path.exists(lookup_pickle) and os.path.exists(oov_pickle):
            logger.info("Directly loading pickled counts.")
            with open(lookup_pickle, "rb") as lp:
                self.lookups = pickle.load(lp)
            with open(oov_pickle, "rb") as op:
                self.oovs = pickle.load(op)
            return True
        else:
            return False

    def get_vocab_word(self, word, key):
        if not word:
            return self.oovs[key]

        if word in self.lookups[key]:
            return word
        else:
            return self.oovs[key]

    @classmethod
    def make_arg(cls, text, role):
        if role == "NA":
            return text + "-" + cls.unk_dep
        else:
            return text + "-" + role

    @staticmethod
    def make_predicate(text):
        return text.lower() + "-pred"

    @staticmethod
    def make_fe(frame, fe):
        # Do not use frame,fe format to alleviate sparsity.
        return fe

    def get_arg_entity_rep(self, arg, entity_text):
        # If a specific entity text is provided.
        rep = self.oovs["argument"]

        if entity_text is not None:
            # Use the argument's own text.
            rep = self.get_vocab_word(entity_text, "argument")

            if rep == self.oovs["argument"]:
                # Use the text after hypen.
                if "-" in entity_text:
                    rep = self.get_vocab_word(entity_text.split("-")[-1], "argument")

        arg_text = arg["text"].lower()
        if rep == self.oovs["argument"]:
            # Fall back to use the argument's own text.
            rep = self.get_vocab_word(arg_text, "argument")

            if rep == self.oovs["argument"]:
                if "-" in arg_text:
                    rep = self.get_vocab_word(arg_text.split("-")[-1], "argument")

        if rep == self.oovs["argument"]:
            # Fall back to NER tag.
            if "ner" in arg:
                rep = arg["ner"]
        return rep

    @classmethod
    def get_unk_arg_rep(cls):
        # This will create a full unknown argument, try to back off to
        #  a partial unknown argument if possible.
        return cls.make_arg(cls.unk_arg_word, cls.unk_dep)

    @classmethod
    def get_unk_arg_with_dep(cls, dep):
        """Return a backoff version of the representation by using the
        actual dep, but unk_arg

        Args:
            dep
        """
        return cls.make_arg(cls.unk_arg_word, dep)

    @classmethod
    def get_arg_rep_no_dep(cls, entity_rep):
        """Return the backoff version of the argument representation by using
        the unk_dep, but the actual entity.

        Args:
          entity_rep:

        Returns:


        """
        return cls.make_arg(entity_rep, cls.unk_dep)

    def get_arg_rep(self, dep, entity_rep):
        if dep.startswith("prep"):
            dep = self.get_vocab_word(dep, "preposition")
        arg_rep = self.make_arg(entity_rep, dep)
        return arg_rep

    def get_pred_rep(self, event):
        """
        Take the predicates, and get the vocab index for it. This will first
         use the predicate itself, if not found, it will try to use the verb
         form.

        :param event:
        :return:
        """
        pred = self.get_vocab_word(event["predicate"], "predicate")

        if pred == self.oovs["predicate"]:
            # Try to see if the verb form help.
            if "verb_form" in event:
                pred = self.get_vocab_word(event["verb_form"], "predicate")
        return self.make_predicate(pred)

    def get_fe_rep(self, frame_name, fe_role):
        # return self.make_fe(frame_name, fe_role)
        return self.get_vocab_word(self.make_fe(frame_name, fe_role), "fe")

    @staticmethod
    def filter_by_count(counter, min_count):
        return [
            (key, count) for key, count in counter.most_common() if count >= min_count
        ]

    def filter_vocab(
        self,
        vocab_counters,
        top_num_prep=150,
        min_token_count=500,
        min_fe_count=50,
        min_frame_count=5,
    ):
        filtered_vocab = {
            "predicate_min_%d"
            % min_token_count: self.filter_by_count(
                vocab_counters["predicate"], min_token_count
            ),
            "argument_min_%d"
            % min_token_count: self.filter_by_count(
                vocab_counters["argument"], min_token_count
            ),
            "preposition_top_%d"
            % top_num_prep: vocab_counters["preposition"].most_common(top_num_prep),
            "fe_min_%d"
            % min_fe_count: self.filter_by_count(vocab_counters["fe"], min_fe_count),
            "frame_min_%d"
            % min_frame_count: self.filter_by_count(
                vocab_counters["frame"], min_frame_count
            ),
        }

        for key, counts in filtered_vocab.items():
            # Use the base key name for the vocabulary, not including the
            # cutoff, (i.e. predicate_min_50 -> predicate)
            name = key.split("_")[0]

            # Put oov token as a token int he vocab file.
            oov = "unk_" + name
            counts.insert(0, (oov, 0))

            self.lookups[name] = {}
            self.oovs[name] = oov

            index = 0
            for term, _ in counts:
                self.lookups[name][term] = index
                index += 1
        return filtered_vocab

    def get_vocab_count(self, data_path):
        vocab_counters = defaultdict(Counter)

        doc_count = 0
        event_count = 0

        with gzip.open(data_path) as data:
            for line in data:
                doc_info = json.loads(line)

                for event in doc_info["events"]:
                    event_count += 1

                    predicate = event["predicate"]
                    vocab_counters["predicate"][predicate] += 1

                    frame = event["frame"]
                    if not frame == "NA":
                        vocab_counters["frame"][frame] += 1

                    for arg in event["arguments"]:
                        fe_name = arg["feName"]
                        syn_role = arg["dep"]
                        arg_text = arg["text"].lower()

                        vocab_counters["argument"][arg_text] += 1

                        if not fe_name == "NA":
                            vocab_counters["fe"][
                                self.make_fe(event["frame"], fe_name)
                            ] += 1

                        if syn_role.startswith("prep"):
                            vocab_counters["preposition"][syn_role] += 1

                doc_count += 1
                if doc_count % 1000 == 0:
                    print(
                        "\rCounted vocab for {} events in "
                        "{} docs.".format(event_count, doc_count),
                        end="",
                    )

        return vocab_counters


class EmbbedingVocab:
    def __init__(self, vocab_file, with_padding=False, extras=None):
        self.vocab_file = vocab_file
        self.vocab = {}
        self.tf = []
        self.extras = []
        self.pad = "__PADDING__"
        self.padded = False

        if with_padding:
            # Paddings should be at 0.
            self.padded = True
            self.vocab[self.pad] = 0
            self.tf.append(0)

        if extras:
            for name in extras:
                self.add_extra(name)

        self.__read_vocab()

    @staticmethod
    def with_extras(vocab_file):
        """
        Create a EmbeddingVocab with unknown word slots and padding slot.
        Args:
            vocab_file:

        Returns:

        """

        return EmbbedingVocab(
            vocab_file,
            True,
            [
                TypedEventVocab.unk_frame,
                TypedEventVocab.unk_fe,
                TypedEventVocab.get_unk_arg_rep(),
                TypedEventVocab.unobserved_arg,
                TypedEventVocab.unobserved_fe,
                TypedEventVocab.ghost,
            ],
        )

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
            logger.info(
                f"Extra {name} already exist in vocabulary "
                f"at index {self.vocab[name]}"
            )
            return self.vocab[name]
        else:
            self.extras.append(name)
            extra_index = len(self.vocab)
            self.vocab[name] = extra_index
            self.tf.append(0)

            logger.info(
                f"Adding {name} as extra dimension {extra_index} "
                f"to {self.vocab_file}"
            )

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


def create_sentences(
    doc,
    event_vocab,
    output_path,
    include_frame=False,
    use_simple_dep=False,
    prop_arg_only=False,
):
    if include_frame:
        print("Adding frames to sentences.")

    doc_count = 0
    event_count = 0

    with gzip.open(doc) as data, gzip.open(output_path, "w") as out:
        for line in data:
            try:
                doc_info = json.loads(line)
            except JSONDecodeError:
                continue

            sentence = []

            represent_by_id = {}
            for entity in doc_info["entities"]:
                eid = entity["entityId"]
                represent = entity["representEntityHead"]
                represent_by_id[eid] = represent

            for event in doc_info["events"]:
                event_count += 1

                sentence.append(event_vocab.get_pred_rep(event))

                if include_frame and not event["frame"] == "NA":
                    frame = event_vocab.get_vocab_word(event["frame"], "frame")
                    sentence.append(frame)

                for arg in event["arguments"]:
                    dep = arg["dep"]

                    if (
                        arg["argStart"] == event["predicateStart"]
                        and arg["argEnd"] == event["predicateEnd"]
                    ):
                        dep = "root"

                    if use_simple_dep:
                        dep = get_simple_dep(dep)

                    if prop_arg_only and not is_propbank_dep(dep):
                        continue

                    sentence.append(
                        event_vocab.get_arg_rep(
                            dep, event_vocab.get_arg_entity_rep(arg, None)
                        )
                    )

                    if include_frame and not arg["feName"] == "NA":
                        fe = event_vocab.get_fe_rep(frame, arg["feName"])
                        if not fe == event_vocab.oovs["fe"]:
                            sentence.append(fe)

                    if "NA" in sentence:
                        pdb.set_trace()

            doc_count += 1

            out.write(str.encode(" ".join(sentence) + "\n"))

            if event_count % 1000 == 0:
                print(
                    "\rCreated sentences for {} documents, "
                    "{} events.".format(doc_count, event_count),
                    end="",
                )

    print(
        "\rCreated sentences for {} documents, "
        "{} events.\n".format(doc_count, event_count),
        end="",
    )


def write_sentences(
    sent_out, event_data, event_vocab, include_frame, simple_dep, prop_arg
):
    if not os.path.exists(sent_out):
        os.makedirs(sent_out)

    fname = "sent_with_frames.gz" if include_frame else "sent_pred_only.gz"

    out = os.path.join(sent_out, fname)
    if not os.path.exists(out):
        create_sentences(
            event_data,
            event_vocab,
            out,
            include_frame=include_frame,
            use_simple_dep=simple_dep,
            prop_arg_only=prop_arg,
        )
    else:
        logger.info(f"Will not overwrite {out}")


def main(event_data, vocab_dir, sent_out, prop_arg):
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    event_vocab = TypedEventVocab(vocab_dir, event_data=event_data)
    logger.info("Done loading vocabulary.")

    # The 3 boolean are : include_frame,simple_dep, prop_arg

    if prop_arg:
        # For propbank style training.
        logger.info("Creating event sentences in propbank style")

        # Include frame or not version for propbank, but always use simple dep
        # and propbank style arguments.
        write_sentences(sent_out, event_data, event_vocab, False, True, True)
        write_sentences(sent_out, event_data, event_vocab, True, True, True)
    else:
        # For framenet style training.
        logger.info("Creating event sentences in FrameNet style")

        # Include frame or not version for framenet, but always use complex dep
        # and framenet style arguments.
        write_sentences(sent_out, event_data, event_vocab, True, False, False)
        write_sentences(sent_out, event_data, event_vocab, False, False, False)


if __name__ == "__main__":
    parser = util.OptionPerLineParser(
        description="Event Vocabulary.", fromfile_prefix_chars="@"
    )
    parser.add_argument("--vocab_dir", type=str, help="Vocabulary directory.")
    parser.add_argument("--input_data", type=str, help="Input data.")
    parser.add_argument("--sent_out", type=str, help="Sentence out dir.")
    parser.add_argument(
        "--prop_arg", action="store_true", help="Propbank arg only.", default=False
    )

    util.set_basic_log()

    args = parser.parse_args()
    main(args.input_data, args.vocab_dir, args.sent_out, args.prop_arg)
