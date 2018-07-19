from traitlets.config import Configurable
from event.arguments.params import ModelPara
from event.arguments.arg_models import EventPairCompositionModel
from traitlets import (
    Unicode,
    Integer
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
from event import torch_util
import pickle


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)

        self.para = ModelPara(**kwargs)
        self.resources = Resources(**kwargs)

        self.debug_dir = kwargs['debug_dir']

        self.device = torch.device(
            "cuda" if self.para.use_gpu and torch.cuda.is_available()
            else "cpu"
        )

        self.model = EventPairCompositionModel(
            self.para, self.resources).to(self.device)

        logging.info("Initialize model")
        print(self.model)

        self.nb_epochs = self.para.nb_epochs

        self.reader = HashedClozeReader(self.resources,
                                        self.para.batch_size,
                                        self.para.multi_context,
                                        self.para.max_events,
                                        self.para.max_cloze)

    def __dump_bug(self, key, obj):
        with open(os.path.join(self.debug_dir, key + '.pickle')) as out:
            pickle.dump(out, obj)

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

                # ones = torch.ones(scores.shape).to(self.device)
                # zeros = torch.zeros(scores.shape).to(self.device)

                ones = torch.ones(1).to(self.device)
                zeros = torch.zeros(1).to(self.device)

                if not ((scores <= ones) & (scores >= zeros)).all():
                    logging.error("Scores not in [0,1] range")
                    print(scores)
                    return None

                # TODO: got error, Assertion `input >= 0. && input <= 1.` failed
                v_loss = F.binary_cross_entropy(scores, gold)
                total_loss += v_loss
            return total_loss
        elif self.para.loss == 'pairwise_letor':
            raise NotImplementedError

    def _get_loss(self, batch_instance, batch_info):
        # print("Before loss")
        # torch_util.show_tensors()
        # torch_util.gpu_mem_report()

        print("Compute gold")
        correct_coh = self.model(
            batch_instance['gold'],
            batch_info,
        )
        print("Compute cross")
        cross_coh = self.model(
            batch_instance['cross'],
            batch_info,
        )
        print("Compute inside")
        inside_coh = self.model(
            batch_instance['inside'],
            batch_info,
        )

        outputs = [correct_coh, cross_coh, inside_coh]
        labels = [1, 0, 0]

        loss = self.__loss(labels, outputs)

        # print("After loss")
        # torch_util.show_tensors()
        # torch_util.gpu_mem_report()

        return loss

    def __check_point(self, check_point_out):
        logging.info("Dumping to checkpoint: {}".format(check_point_out))
        torch.save(self.model.state_dict(), check_point_out)

    def __resume(self, check_point_out):
        self.model.load_state_dict(torch.load(check_point_out))

    def validation(self, generator):
        dev_loss = 0
        num_batches = 0
        for batch_instance, batch_info in generator:
            loss = self._get_loss(batch_instance, batch_info)
            if not loss:
                raise ValueError('Error in computing loss.')
            dev_loss += loss.item()
            num_batches += 1

        logging.info("Validate with [%d] batches." % num_batches)

        return dev_loss

    def train(self, train_in, validation_size=None, validation_in=None,
              model_out_dir=None, debug_out=None):
        logging.info("Training with data [%s]", train_in)

        if validation_in:
            logging.info("Validation with data [%s]", validation_in)
        elif validation_size:
            logging.info(
                "Will use first few [%d] for validation." % validation_size)
        else:
            logging.error("No validaiton!")

        if model_out_dir:
            logging.info("Model out directory is [%s]", model_out_dir)
            if not os.path.exists(model_out_dir):
                os.makedirs(model_out_dir)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        batch_count = 0
        total_loss = 0

        recent_loss = 0

        dev_instances = []

        previous_dev_loss = math.inf
        worse = 0

        log_freq = 100

        for epoch in range(self.nb_epochs):
            with smart_open(train_in) as train_data:
                for batch_instance, batch_info in self.reader.read_cloze_batch(
                        train_data, from_line=validation_size):

                    loss = self._get_loss(batch_instance, batch_info)
                    if not loss:
                        # with open(
                        #         os.path.join(
                        #             debug_out,
                        #             'batch_instance.pickle')) as instace_bug:
                        #     pickle.dump(instace_bug, batch_instance)
                        # with open(
                        #     os.path.join(
                        #         debug_out,
                        #         os.path.join(debug_out, 'batch_info.pickle')
                        #     ) as info_bug:
                        #
                        # ):
                        #     pickle.dumps(
                        #         os.path.join(debug_out, 'batch_info.pickle'),
                        #         batch_info)
                        self.__dump_bug('batch_instance', batch_instance)
                        self.__dump_bug('batcH_info', batch_info)
                        raise ValueError('Error in computing loss.')

                    loss_val = loss.item()
                    total_loss += loss_val
                    recent_loss += loss_val

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_count += 1

                    if not batch_count % log_freq:
                        logging.info(
                            "Batch {} ({} instances), "
                            "recent avg. loss {}, overall avg. loss {:.5f}".
                                format(batch_count,
                                       batch_count * self.reader.batch_size,
                                       recent_loss / log_freq,
                                       total_loss / batch_count)
                        )
                        # torch_util.gpu_mem_report()
                        recent_loss = 0

            logging.info("Conducting validation.")
            dev_generator = None
            if validation_in:
                with smart_open(validation_in) as dev_data:
                    dev_generator = self.reader.read_cloze_batch(dev_data)

            if validation_size:
                with smart_open(train_in) as train_data:
                    dev_generator = self.reader.read_cloze_batch(
                        train_data, until_line=validation_size)

            dev_loss = self.validation(dev_generator)

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
        validation_size = Integer(help='validation size').tag(config=True)
        model_name = Unicode(help='model name').tag(config=True)
        debug_dir = Unicode(help='Debug output').tag(config=True)


    from event.util import set_basic_log
    from event.util import load_command_line_config

    set_basic_log()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logging.info("Started the runner.")

    cl_conf = load_command_line_config(sys.argv[2:])
    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    conf.merge(cl_conf)

    basic_para = Basic(config=conf)

    model = ArgRunner(config=conf, debug_dir=basic_para.debug_dir)

    base = os.environ['implicit_corpus']
    model_out = os.path.join(base, 'models', basic_para.model_name)

    if basic_para.debug_dir and not os.path.exists(basic_para.debug_dir):
        os.makedirs(basic_para.debug_dir)

    model.train(basic_para.train_in, basic_para.validation_size,
                basic_para.valid_in, model_out)

    # pr.disable()
    # pr.dump_stats('../profile.dump')
    # pr.print_stats(sort='time')
