from event.io.readers import TaggedMentionReader
from event.util import set_basic_log
from event.mention.models.detectors import TextCNN
import logging


class DetectionRunner:
    def __init__(self, config):
        self.init_model(config)

    def init_model(self, config):
        self.model_type = config.model_type
        self.model = TextCNN(config)

    def train(self, reader):
        for token_ids, tag_ids in reader.read():
            print(token_ids)
            print(tag_ids)

            import sys
            sys.stdin.readline()

            self.model.train()

    def test(self, reader):
        reader.hash()
        self.model.test()


def main(config):
    reader = TaggedMentionReader(config.train_data_files, config)
    reader.hash()

    detector = DetectionRunner(config)
    detector.train(reader)
    detector.test(reader)


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Event Mention Detector.',
                                 fromfile_prefix_chars='@')

    parser.add_argument('--word_embedding_dim', type=float, default=300)
    parser.add_argument('--position_embedding_dim', type=float, default=50)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')

    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--train_data_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--test_data_files',
                        type=lambda s: [item for item in s.split(',')])

    parser.add_argument('--format', type=str, default="conllu")
    parser.add_argument('--no_punct', type=bool, default=False)
    parser.add_argument('--no_sentence', type=bool, default=False)

    parser.add_argument('--model_type', type=str, default="cnn")

    arguments = parser.parse_args()

    set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
