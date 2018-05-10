from traitlets.config import Configurable
from event.arguments.params import ModelPara
from event.arguments.arg_models import EventPairCompositionModel
from traitlets import (
    Unicode
)
from traitlets.config.loader import PyFileConfigLoader
import torch

import logging
import sys

from event.io.readers import HashedClozeReader
from event.arguments.loss import cross_entropy

from event.util import smart_open

class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)
        self.para = ModelPara(**kwargs)
        self.model = EventPairCompositionModel(self.para)

        self.nb_epochs = self.para.nb_epochs
        self.criterion = cross_entropy

        self.reader = HashedClozeReader()

    def train(self, train_in, validation_in=None, model_out=None):
        logging.info("Training with data [%s]", train_in)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.nb_epochs):
            with smart_open(train_in) as train_data:
                for cloze_task in self.reader.read_clozes(train_data):
                    correct, cross_instance, inside_instance = cloze_task

                    print(event_index, cloze_role, answer, wrong)

                    output = self.model(self.process_data(line))
                    loss = self.criterion(output, label)
                    loss.backward()
                    optimizer.step()

    def process_data(self, line):
        print(line)
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
