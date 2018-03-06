from io import TaggedMentionReader


def train(reader):
    for token_ids, tag_ids in reader.read():
        pass


def test(reader):
    reader.hash()


def main(config):
    reader = TaggedMentionReader(config)
    reader.hash()
    train(reader)
    test(reader)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Event Mention Detector.',
                                     fromfile_prefix_chars='@')

    parser.add_argument('--config', help='JSON file of configs',
                        type=argparse.FileType('r'))

    parser.add_argument('--word_embedding_dim', type=float, default=300)
    parser.add_argument('--position_embedding_dim', type=float, default=50)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')

    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--data_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--format', type=str, default="conllu")
    parser.add_argument('--no_punct', type=bool, default=False)
    parser.add_argument('--no_sentence', type=bool, default=False)

    arguments = parser.parse_args()

    if arguments.infile:
        config = json.load(arguments.infile[0])

    print(arguments)

    main(arguments)
