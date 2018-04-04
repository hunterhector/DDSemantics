from traitlets.config import Configurable
from event.arguments.params import ModelPara
from event.arguments.arg_models import EventPairCompositionModel
from traitlets import (
    Unicode
)
from traitlets.config.loader import PyFileConfigLoader
import torch

import logging


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)
        self.para = ModelPara(**kwargs)
        self.model = EventPairCompositionModel(self.para)

        self.nb_epochs = self.para.nb_epochs

    def train(self, train_in, validation_in=None, model_out=None):
        logging.info("Training with data [%s]", train_in)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        print(self.model.parameters)

        for epoch in range(self.nb_epochs):
            for line in train_in:
                output = self.model(self.process_data(line))
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

    def process_data(self, line):
        pass


if __name__ == '__main__':
    class Basic(Configurable):
        train_in = Unicode(help='training data').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        model_out = Unicode(help='model dump out name').tag(config=True)


    import sys

    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    basic_para = Basic(config=conf)

    model = ArgRunner(config=conf)
    model.train(basic_para.train_in, basic_para.valid_in, basic_para.model_out)
