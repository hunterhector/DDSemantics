from event.io.readers import (
    ConllUReader,
    Vocab
)
from event.io.csr import CSR
from event.mention.models.detectors import (
    TextCNN,
    FrameMappingDetector
)
import logging
import torch
import torch.nn.functional as F
import os
import json
from event import util


class DetectionRunner:
    def __init__(self, config, token_vocab, tag_vocab):
        self.model_dir = config.model_dir
        self.model_name = config.model_name
        self.trainable = True

        self.init_model(config, token_vocab, tag_vocab)

    def init_model(self, config, token_vocab, tag_vocab):
        if self.model_name == 'cnn':
            self.model = TextCNN(config, tag_vocab.vocab_size(),
                                 token_vocab.vocab_size())
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.model_name == 'frame':
            self.model = FrameMappingDetector(config, token_vocab)
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

        train_reader.tag_vocab.fix()

    def eval(self, dev_reader):
        return 0

    def save(self):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, self.model_name)
        torch.save(self.model, path)

    def predict(self, test_reader, csr):
        for data in test_reader.read_window():
            tokens, tags, features, l_word_meta, meta = data

            # Found the center lemma's type and possible arguments in
            # the window.
            event_type, args = self.model.predict(data)

            center = int(len(l_word_meta) / 2)

            token, span = l_word_meta[center]

            if not event_type == self.model.unknown_type:
                extent_span = [span[0], span[1]]
                for role, (index, entity_type) in args.items():
                    a_token, a_span = l_word_meta[index]
                    if a_span[0] < extent_span[0]:
                        extent_span[0] = a_span[0]
                    if a_span[1] > extent_span[1]:
                        extent_span[1] = a_span[1]

                event_info = csr.add_event_mention(span, token, 'aida',
                                                   event_type, component='aida')

                if event_info:
                    event_id, interp = event_info
                    for role, (index, entity_type) in args.items():
                        a_token, a_span = l_word_meta[index]
                        entity_id = csr.add_entity_mention(
                            a_span, a_token, 'aida', entity_type
                        )

                        if entity_id:
                            csr.add_event_arg(interp, event_id, entity_id,
                                              'aida', role, 'Implicit')


def main(config):
    token_vocab = Vocab(config.experiment_folder, 'tokens',
                        embedding_path=config.word_embedding,
                        emb_dim=config.word_embedding_dim)

    tag_vocab = Vocab(config.experiment_folder, 'tag',
                      embedding_path=config.tag_list)

    train_reader = ConllUReader(config.train_files, config, token_vocab,
                                tag_vocab, config.language)
    dev_reader = ConllUReader(config.dev_files, config, token_vocab,
                              train_reader.tag_vocab, config.language)
    detector = DetectionRunner(config, token_vocab, tag_vocab)
    detector.train(train_reader, dev_reader)

    #     def __init__(self, component_name, run_id, out_path):
    res_collector = CSR('Event_hector_frames', 1, config.output, 'belcat')

    test_reader = ConllUReader(config.test_files, config, token_vocab,
                               train_reader.tag_vocab)

    detector.predict(test_reader, res_collector)

    res_collector.write()


if __name__ == '__main__':
    from event import util

    parser = util.basic_parser()

    arguments = parser.parse_args()

    util.set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
