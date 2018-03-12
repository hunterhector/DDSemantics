from event.io.readers import TaggedMentionReader
from event.util import set_basic_log
from event.mention.models.detectors import (
    TextCNN,
    FrameMappingDetector
)
import logging
import torch
import torch.nn.functional as F
import os


class DetectionRunner:
    def __init__(self, config, num_classes, vocab_size):
        self.model_dir = config.model_dir
        self.model_name = config.model_name
        self.trainable = True

        self.init_model(config, num_classes, vocab_size)

    def init_model(self, config, num_classes, vocab_size):
        if self.model_name == 'cnn':
            self.model = TextCNN(config, num_classes, vocab_size)
        elif self.model_name == 'frame':
            self.model = FrameMappingDetector(config)
            self.trainable = False

    def train(self, train_reader, dev_reader):
        if not self.trainable:
            return

        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        epoch = 10
        step = 0
        best_step = 0
        log_every_k = 10

        early_patience = 10

        best_res = 0

        for epoch in range(epoch):
            input, labels = train_reader.read_batch()
            optimizer.zero_grad()

            print("Batch:")
            print(input)
            print(labels)
            logits = self.model(input)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            optimizer.step()
            step += 1

            # Eval on dev.
            if not step % log_every_k:
                dev_res = self.eval(dev_reader)

                if dev_res > best_res:
                    best_res = dev_res
                    best_step = step
                    self.save()
                else:
                    if step - best_step > early_patience:
                        logging.info(
                            "Early stop with patience %d." % early_patience
                        )

    def eval(self, dev_reader):
        return 0

    def save(self):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, self.model_name)
        torch.save(self.model, path)

    def predict(self, test_reader):
        test_reader.hash()
        for data in test_reader.read_window():
            tokens, tags = data
            print(test_reader.reveal_tokens(tokens))
            print(test_reader.reveal_tags(tags))
            self.model.predict(data)


def main(config):
    train_reader = TaggedMentionReader(config.train_files, config)

    dev_reader = TaggedMentionReader(config.dev_files, config,
                                     train_reader.token_dict(),
                                     train_reader.tag_dict())

    test_reader = TaggedMentionReader(config.test_files, config,
                                      train_reader.token_dict(),
                                      train_reader.tag_dict())

    detector = DetectionRunner(config, train_reader.num_classes(),
                               train_reader.vocab_size())

    detector.train(train_reader, dev_reader)
    detector.predict(test_reader)




if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Event Mention Detector.',
                                 fromfile_prefix_chars='@')

    parser.add_argument('--model_name', type=str, default='cnn')

    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--model_dir', type=str)

    parser.add_argument('--train_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--dev_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--test_files',
                        type=lambda s: [item for item in s.split(',')])

    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--position_embedding_dim', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')
    parser.add_argument('--window_sizes', default='2,3,4,5',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--filter_num', default=100, type=int,
                        help='Number of filters for each type.')
    parser.add_argument('--fix_embedding', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=50)

    parser.add_argument('--format', type=str, default="conllu")
    parser.add_argument('--no_punct', type=bool, default=False)
    parser.add_argument('--no_sentence', type=bool, default=False)

    parser.add_argument('--frame_lexicon', type=str, default=False)

    arguments = parser.parse_args()

    set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
