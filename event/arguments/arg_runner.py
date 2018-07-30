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

from smart_open import smart_open
from event.arguments.resources import Resources
import math
import os
from event import torch_util
import pickle
import shutil


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)

        self.para = ModelPara(**kwargs)
        self.resources = Resources(**kwargs)

        self.model_dir = kwargs['model_dir']
        self.debug_dir = kwargs['debug_dir']

        self.checkpoint_name = 'checkpoint.pth'

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

                # TODO: Got NaN in scores
                if torch.isnan(scores).any():
                    logging.error("NaN in scores")
                    print(scores)
                    return None

                v_loss = F.binary_cross_entropy(scores, gold)
                total_loss += v_loss
            return total_loss
        elif self.para.loss == 'pairwise_letor':
            raise NotImplementedError

    def _get_loss(self, batch_instance, batch_info):
        # print("Before loss")
        # torch_util.show_tensors()
        # torch_util.gpu_mem_report()

        correct_coh = self.model(
            batch_instance['gold'],
            batch_info,
        )
        cross_coh = self.model(
            batch_instance['cross'],
            batch_info,
        )

        inside_coh = self.model(
            batch_instance['inside'],
            batch_info,
        )

        outputs = [correct_coh, cross_coh, inside_coh]
        labels = [1, 0, 0]

        loss = self.__loss(labels, outputs)

        return loss

    def __dump_stuff(self, key, obj):
        logging.info("Saving object: {}.".format(key))
        with open(os.path.join(self.debug_dir, key + '.pickle'), 'wb') as out:
            pickle.dump(obj, out)

    def __load_stuff(self, key):
        logging.info("Loading object: {}.".format(key))
        with open(os.path.join(self.debug_dir, key + '.pickle'), 'rb') as fin:
            return pickle.load(fin)

    def __save_checkpoint(self, state, is_best):
        check_path = os.path.join(self.model_dir, self.checkpoint_name)
        best_path = os.path.join(self.model_dir, 'model_best.pth')

        logging.info("Saving model at {}".format(check_path))
        torch.save(state, check_path)

        if is_best:
            logging.info("Saving model as best {}".format(best_path))
            shutil.copyfile(check_path, best_path)

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

    def debug(self):
        checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
        if os.path.isfile(checkpoint_path):
            logging.info("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])

        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(param)
                print(name)
                print("NaN in ", name)

        batch_instance = self.__load_stuff('batch_instance')
        batch_info = self.__load_stuff('batch_info')

        self._get_loss(batch_instance, batch_info)

    def train(self, train_in, validation_size=None, validation_in=None,
              model_out_dir=None, resume=False):
        logging.info("Training with data from [%s]", train_in)

        if validation_in:
            logging.info("Validation with data from [%s]", validation_in)
        elif validation_size:
            logging.info(
                "Will use first few [%d] for validation." % validation_size)
        else:
            logging.error("No validation!")

        if model_out_dir:
            logging.info("Model out directory is [%s]", model_out_dir)
            if not os.path.exists(model_out_dir):
                os.makedirs(model_out_dir)

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        start_epoch = 0
        batch_count = 0
        instance_count = 0
        best_loss = math.inf

        if resume:
            checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
            if os.path.isfile(checkpoint_path):
                logging.info("Loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)

                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                logging.info(
                    "No model to resume at '{}', starting from scratch.".format(
                        checkpoint_path)
                )

        total_loss = 0
        recent_loss = 0

        dev_instances = []

        previous_dev_loss = math.inf
        worse = 0

        log_freq = 100

        def data_gen(data_path):
            if os.path.isdir(data_path):
                for f in os.listdir(data_path):
                    logging.info("Reading from {}".format(f))
                    with smart_open(os.path.join(data_path, f)) as fin:
                        for line in fin:
                            yield line
            else:
                with smart_open(data_path) as fin:
                    logging.info("Reading from {}".format(data_path))
                    for line in fin:
                        yield line

        for epoch in range(start_epoch, self.nb_epochs):
            logging.info("Starting epoch {}.".format(epoch))

            for batch_data in self.reader.read_cloze_batch(
                    data_gen(train_in), from_line=validation_size):

                batch_instance, batch_info, n_instance = batch_data

                loss = self._get_loss(batch_instance, batch_info)
                if not loss:
                    self.__save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict(),
                    }, False)

                    for name, weight in self.model.named_parameters():
                        if name.startswith('event_to_var_layer'):
                            print(name, weight)

                    self.__dump_stuff('batch_instance', batch_instance)
                    self.__dump_stuff('batch_info', batch_info)

                    raise ValueError('Error in computing loss.')

                loss_val = loss.item()
                total_loss += loss_val
                recent_loss += loss_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1
                instance_count += n_instance

                if not batch_count % log_freq:
                    logging.info(
                        "Epoch {}, Batch {} ({} instances), "
                        "recent avg. loss {}, overall avg. loss {:.5f}".format(
                            epoch, batch_count, instance_count,
                            recent_loss / log_freq,
                            total_loss / batch_count)
                    )

                    for name, weight in self.model.named_parameters():
                        if name.startswith('event_to_var_layer'):
                            print(name, weight)

                    recent_loss = 0

            logging.info("Conducting validation.")

            dev_generator = None

            if validation_in:
                dev_generator = self.reader.read_cloze_batch(
                    data_gen(validation_in))

            if validation_size:
                dev_generator = self.reader.read_dir(
                    data_gen(train_in), until_line=validation_size)

            dev_loss = self.validation(dev_generator)

            is_best = False
            if not best_loss or dev_loss < best_loss:
                best_loss = dev_loss
                is_best = True

            logging.info(
                "Finished epoch {epoch:d}, avg. training loss {loss:.4f}, "
                "validation loss {dev_loss:.4f}{best:s}".format(
                    epoch=epoch, loss=total_loss / batch_count,
                    dev_loss=dev_loss / len(dev_instances),
                    best=', is current best.' if is_best else '.'
                ))

            if dev_loss < previous_dev_loss:
                previous_dev_loss = dev_loss
                worse = 0
                self.__save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

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
        train_in = Unicode(help='training data directory').tag(config=True)
        test_in = Unicode(help='testing data').tag(config=True)
        test_out = Unicode(help='test res').tag(config=True)
        valid_in = Unicode(help='validation in').tag(config=True)
        validation_size = Integer(help='validation size').tag(config=True)
        debug_dir = Unicode(help='Debug output').tag(config=True)
        model_name = Unicode(help='model name', default_value='basic').tag(
            config=True)
        model_dir = Unicode(help='model directory').tag(config=True)


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

    model = ArgRunner(
        config=conf,
        model_dir=basic_para.model_dir,
        debug_dir=basic_para.debug_dir
    )

    if basic_para.debug_dir and not os.path.exists(basic_para.debug_dir):
        os.makedirs(basic_para.debug_dir)

    model.train(
        basic_para.train_in,
        validation_size=basic_para.validation_size,
        validation_in=basic_para.valid_in,
        resume=True
    )
