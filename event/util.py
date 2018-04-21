import argparse
import logging
import sys
import os


class OptionPerLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []
        return arg_line.split()


def ensure_dir(p):
    parent = os.path.dirname(p)
    if not os.path.exists(parent):
        os.makedirs(parent)


def tokens_to_sent(tokens, sent_start):
    sent = ""

    for token, span in tokens:
        if span[0] > len(sent) + sent_start:
            padding = ' ' * (span[0] - len(sent) - sent_start)
            sent += padding
        sent += token
    return sent


def basic_parser():
    parser = OptionPerLineParser(description='Event Mention Detector.',
                                 fromfile_prefix_chars='@')

    parser.add_argument('--model_name', type=str)

    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--model_dir', type=str)

    parser.add_argument('--train_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--dev_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--test_folder', type=str)

    parser.add_argument('--output', type=str)

    parser.add_argument('--word_embedding', type=str,
                        help='Word embedding path')
    parser.add_argument('--word_embedding_dim', type=int, default=300)

    parser.add_argument('--position_embedding_dim', type=int, default=50)

    parser.add_argument('--tag_list', type=str,
                        help='Frame embedding path')
    parser.add_argument('--tag_embedding_dim', type=int, default=50)

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')
    parser.add_argument('--context_size', default=30, type=int)
    parser.add_argument('--window_sizes', default='2,3,4,5',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--filter_num', default=100, type=int,
                        help='Number of filters for each type.')
    parser.add_argument('--fix_embedding', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=50)

    parser.add_argument('--format', type=str, default="conllu")
    parser.add_argument('--no_punct', type=bool, default=False)
    parser.add_argument('--no_sentence', type=bool, default=False)

    # Frame based detector.
    parser.add_argument('--frame_lexicon', type=str, help='Frame lexicon path')
    parser.add_argument('--event_list', help='Lexicon for events', type=str)
    parser.add_argument('--entity_list', help='Lexicon for entities', type=str)
    parser.add_argument('--relation_list', help='Lexicon for relations',
                        type=str)

    parser.add_argument('--language', help='Language code', type=str)

    return parser


def set_basic_log(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
