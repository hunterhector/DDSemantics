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
from event.arguments.resources import Resources

import pprint


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)
        self.para = ModelPara(**kwargs)
        self.resources = Resources(**kwargs)
        self.model = EventPairCompositionModel(self.para, self.resources)

        self.nb_epochs = self.para.nb_epochs
        self.criterion = cross_entropy

        self.batch_size = self.para.batch_size

        self.reader = HashedClozeReader(self.resources.event_vocab_path,
                                        self.para.multi_context,
                                        self.para.max_events)

    def train(self, train_in, validation_in=None, model_out=None):
        logging.info("Training with data [%s]", train_in)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.nb_epochs):
            instance_count = 0
            batch_data = {
                'context': [],
                'gold_data': [],
                'cross_data': [],
                'inside_data': [],
            }
            with smart_open(train_in) as train_data:
                for cloze_task in self.reader.read_clozes(train_data):
                    print("Cloze task")
                    pprint.PrettyPrinter(indent=2).pprint(cloze_task)
                    input("A cloze task.")

                    batch_data['context'].append(
                        cloze_task['context']
                    )

                    batch_data['gold_data'].append([
                        cloze_task['gold'],
                        cloze_task['gold_features'],
                        cloze_task['gold_distance'],
                    ])

                    batch_data['cross_data'].append([
                        cloze_task['cross'],
                        cloze_task['cross_features'],
                        cloze_task['cross_distance'],
                    ])

                    batch_data['inside_data'].append([
                        cloze_task['inside'],
                        cloze_task['inside_features'],
                        cloze_task['inside_distance'],
                    ])

                    if len(batch_data) == self.batch_size:
                        correct_coh = self.model(
                            batch_data['context'], batch_data['gold_data']
                        )
                        cross_coh = self.model(
                            batch_data['context'], batch_data['cross_data']
                        )
                        inside_coh = self.model(
                            batch_data['context'], batch_data['inside_data']
                        )

                        optimizer.zero_grad()
                        correct_loss = self.criterion(1, correct_coh)
                        cross_loss = self.criterion(0, cross_coh)
                        inside_loss = self.criterion(0, inside_coh)

                        full_loss = correct_loss + cross_loss + inside_loss

                        full_loss.backward()
                        optimizer.step()

                        instance_count += 1

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
