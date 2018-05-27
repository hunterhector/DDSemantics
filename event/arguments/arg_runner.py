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

from event.io.readers import HashedClozeReader

from event.util import smart_open
from event.arguments.resources import Resources


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)
        self.para = ModelPara(**kwargs)
        self.resources = Resources(**kwargs)
        self.model = EventPairCompositionModel(self.para, self.resources)

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
                    gold = torch.ones(scores.shape)
                else:
                    gold = torch.zeros(scores.shape)

                v_loss = F.binary_cross_entropy(scores, gold)
                total_loss += v_loss
            return total_loss
        elif self.para.loss == 'pairwise_letor':
            raise NotImplementedError

    def train(self, train_in, validation_in=None, model_out=None):
        logging.info("Training with data [%s]", train_in)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.nb_epochs):
            with smart_open(train_in) as train_data:
                for batch_instance, batch_info in self.reader.read_cloze_batch(
                        train_data):
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

                    optimizer.zero_grad()

                    loss = self.__loss(labels, outputs)

                    print("Loss")
                    print(loss)

                    loss.backward()
                    optimizer.step()

    def event_repr(self, l_predicate, l_args):

        pass


if __name__ == '__main__':
    class Basic(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)


    logging.basicConfig(level=logging.INFO)

    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    basic_para = Basic(config=conf)

    model = ArgRunner(config=conf)
    model.train(basic_para.train_in, basic_para.valid_in, basic_para.model_out)
