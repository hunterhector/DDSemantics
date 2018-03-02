from io import TaggedMentionReader


def train(reader):
    reader.hash()
    for token_ids, tag_ids in reader.read():
        pass

def test(reader):
    reader.hash()


def main(args):
    reader = TaggedMentionReader(args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Event Mention Detector.')

    parser.add_argument('--word_embedding_dim', type=float, default=300)
    parser.add_argument('--position_embedding_dim', type=float, default=50)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')

    args = parser.parse_args()

    main(args)
