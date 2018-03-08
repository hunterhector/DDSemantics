from event.io.readers import TaggedMentionReader
from event.util import set_basic_log
from event.mention.models.detectors import TextCNN
import logging
import torch
import torch.nn.fuciontal as F
import os

use_cuda = torch.cuda.is_available()


class DetectionRunner:
    def __init__(self, config):
        self.init_model(config)
        self.model_dir = config.model_dir
        self.model_name = config.model_name

    def init_model(self, config):
        self.model_type = config.model_type
        self.model = TextCNN(config)
        if use_cuda:
            self.model.cuda()

    def train(self, reader, dev_data):
        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        epoch = 10
        step = 0
        best_step = 0
        log_every_k = 10

        early_patience = 10

        best_res = 0

        for epoch in range(epoch):
            input, labels = reader.read_batch()
            optimizer.zero_grad()

            if use_cuda:
                input, labels = input.cuda(), labels.cuda()

            logits = self.model(input)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            optimizer.step()
            step += 1

            # Eval on dev.
            if not step % log_every_k:
                dev_res = self.eval(dev_data)

                if dev_res > best_res:
                    best_res = dev_res
                    best_step = step
                    self.save()
                else:
                    if step - best_step > early_patience:
                        logging.info(
                            "Early stop with patience %d." % early_patience
                        )

    def eval(self, dev_data):
        return 0

    def save(self):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, self.model_name)
        torch.save(self.model, path)

    def test(self, reader):
        reader.hash()
        self.model.test()


def main(config):
    reader = TaggedMentionReader(config.train_data_files, config)
    reader.hash()

    detector = DetectionRunner(config)
    detector.train(reader, )
    detector.test(reader)


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Event Mention Detector.',
                                 fromfile_prefix_chars='@')

    parser.add_argument('--word_embedding_dim', type=float, default=300)
    parser.add_argument('--position_embedding_dim', type=float, default=50)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')

    parser.add_argument('--model_name', type=str)

    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--model_dir', type=str)

    parser.add_argument('--train_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--dev_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--test_files',
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
