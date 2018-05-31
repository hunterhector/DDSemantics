from traitlets.config import Configurable
from event.arguments.params import ModelPara
from event.arguments.arg_models import EventPairCompositionModel
from traitlets import (
    Unicode
)
from traitlets.config.loader import PyFileConfigLoader
import torch
from torch.nn import functional as F

import logging
import sys

from event.arguments.cloze_readers import HashedClozeReader

from event.util import smart_open
from event.arguments.resources import Resources
import math
import os


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)

        self.para = ModelPara(**kwargs)
        self.resources = Resources(**kwargs)

        self.device = torch.device(
            "cuda" if self.para.use_gpu and torch.cuda.is_available()
            else "cpu"
        )

        self.model = EventPairCompositionModel(
            self.para, self.resources).to(self.device)

        logging.info("Initialize model")
        print(self.model)

        self.nb_epochs = self.para.nb_epochs

        self.reader = HashedClozeReader(
            self.resources.event_vocab,
            self.resources.lookups,
            self.resources.oovs,
            self.para.batch_size,
            self.para.multi_context,
            self.para.max_events,
        )

    def _assert(self):
        if self.resources.word_embedding:
            assert self.para.word_vocab_size == \
                   self.resources.word_embedding.shape[0]
        if self.resources.event_embedding:
            assert self.para.event_arg_vocab_size == \
                   self.resources.event_embedding.shape[0]

    def __loss(self, l_label, l_scores):
        if self.para.loss == 'cross_entropy':
            total_loss = 0
            for label, scores in zip(l_label, l_scores):
                if label == 1:
                    gold = torch.ones(scores.shape).to(self.device)
                else:
                    gold = torch.zeros(scores.shape).to(self.device)

                v_loss = F.binary_cross_entropy(scores, gold)
                total_loss += v_loss
            return total_loss
        elif self.para.loss == 'pairwise_letor':
            raise NotImplementedError

    def _get_loss(self, batch_instance, batch_info):
        correct_coh = self.model(
            batch_instance['gold'],
            batch_info,
        )
        cross_coh = self.model(
            batch_instance['cross'],
            batch_info
        )
        inside_coh = self.model(
            batch_instance['inside'],
            batch_info
        )

        outputs = [correct_coh, cross_coh, inside_coh]
        labels = [1, 0, 0]
        loss = self.__loss(labels, outputs)
        return loss

    def __check_point(self, check_point_out):
        logging.info("Dumping to checkpoint: {}".format(check_point_out))
        torch.save(self.model.state_dict(), check_point_out)

    def __resume(self, check_point_out):
        self.model.load_state_dict(torch.load(check_point_out))

    def train(self, train_in, validation_in=None, model_out_dir=None):
        logging.info("Training with data [%s]", train_in)
        logging.info("Validation with data [%s]", validation_in)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        batch_count = 0
        total_loss = 0

        recent_loss = 0

        dev_instances = []

        previous_dev_loss = math.inf
        worse = 0

        with smart_open(validation_in) as dev_data:
            for data in self.reader.read_cloze_batch(dev_data):
                dev_instances.append(data)
        logging.info(
            "Loaded {} validation batches".format(len(dev_instances)))

        for epoch in range(self.nb_epochs):
            # TODO: improve training speed.
            with smart_open(train_in) as train_data:
                for batch_instance, batch_info in self.reader.read_cloze_batch(
                        train_data):
                    loss = self._get_loss(batch_instance, batch_info)

                    loss_val = loss.item()
                    total_loss += loss_val
                    recent_loss += loss_val

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    batch_count += 1

                    if not batch_count % 100:
                        logging.info(
                            "Batch {} ({} instances), "
                            "recent avg. loss {}, overall avg. loss {}".
                                format(batch_count,
                                       batch_count * self.reader.batch_size,
                                       recent_loss / 100,
                                       total_loss / batch_count)
                        )
                        recent_loss = 0

            dev_loss = 0
            for batch_instance, batch_info in dev_instances:
                loss = self._get_loss(batch_instance, batch_info)
                dev_loss += loss.item()

            logging.info(
                "Finished epoch {epoch:d}, avg. training loss {loss:.4f}, "
                "validation loss {dev_loss:.4f}".format(
                    epoch=epoch, loss=total_loss / batch_count,
                    dev_loss=dev_loss / len(dev_instances)
                ))

            if dev_loss < previous_dev_loss:
                previous_dev_loss = dev_loss
                worse = 0
                self.__check_point(
                    os.path.join(model_out_dir, 'current_best.model')
                )
            else:
                worse += 1
                if worse == self.para.early_stop_patience:
                    logging.info(
                        "Dev loss increase from {pre:.4f} to {curr:.4f}, "
                        "stop at Epoch {epoch:d}"
                    ).format(pre=previous_dev_loss, curr=dev_loss, epoch=epoch)
                    break


if __name__ == '__main__':
    class Basic(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        model_out = Unicode(help='model output directory').tag(config=True)


    from event.util import set_basic_log

    set_basic_log()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logging.info("Started the runner.")

    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    basic_para = Basic(config=conf)

    model = ArgRunner(config=conf)
    model.train(basic_para.train_in, basic_para.valid_in, basic_para.model_out)
