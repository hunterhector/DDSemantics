from traitlets.config import Configurable
from event.arguments.impicit_arg_params import ArgModelPara
from event.arguments.arg_models import EventPairCompositionModel
from traitlets import (
    Unicode,
    Integer,
    Bool,
)
import torch
from torch.nn import functional as F

import logging
import sys

from event.arguments.cloze_readers import HashedClozeReader

from smart_open import smart_open
from event.arguments.implicit_arg_resources import ImplicitArgResources
import math
import os
import pickle
import shutil
from pprint import pprint
from itertools import groupby
from operator import itemgetter
from event.arguments.evaluation import ImplicitEval
from event import util
from collections import Counter


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


class NullArgDetector:
    def __init__(self):
        pass

    def get_slot_to_fill(self, doc_info):
        pass


class GoldNullArgDetector(NullArgDetector):
    def __index__(self):
        super(NullArgDetector, self).__init__()

    def get_filling_slot(self, arg_info):
        return arg_info['to_fill']


class ArgRunner(Configurable):

    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)

        self.para = ArgModelPara(**kwargs)
        self.resources = ImplicitArgResources(**kwargs)

        self.model_dir = kwargs['model_dir']
        self.debug_dir = kwargs['debug_dir']

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        logging.info("Model saving directory: " + self.model_dir)

        self.checkpoint_name = 'checkpoint.pth'

        self.best_model_name = 'model_best.pth'

        self.device = torch.device(
            "cuda" if self.para.use_gpu and torch.cuda.is_available()
            else "cpu"
        )

        self.model = EventPairCompositionModel(
            self.para, self.resources).to(self.device)

        logging.info("Initialize model")
        logging.info(str(self.model))

        self.nb_epochs = self.para.nb_epochs

        self.reader = HashedClozeReader(self.resources,
                                        self.para.batch_size,
                                        multi_context=self.para.multi_context,
                                        max_events=self.para.max_events,
                                        max_cloze=self.para.max_cloze)

        # Set up Null Instantiation.
        if self.para.nid_method == 'gold':
            self.nid_detector = GoldNullArgDetector()
        else:
            raise NotImplementedError

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

    def _get_loss(self, batch_instance, batch_common):
        correct_coh = self.model(batch_instance['gold'], batch_common)

        cross_coh = self.model(batch_instance['cross'], batch_common)

        inside_coh = self.model(batch_instance['inside'], batch_common)

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
        num_instances = 0

        for batch_data in generator:
            batch_instance, batch_common, n_instance = batch_data
            loss = self._get_loss(batch_instance, batch_common)
            if not loss:
                raise ValueError('Error in computing loss.')
            dev_loss += loss.item()
            num_batches += 1
            num_instances += 1

        logging.info("Validate with [%d] batches, [%d] instances." % (
            num_batches, num_instances))

        return dev_loss, num_batches, num_instances

    def debug(self):
        checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
        if os.path.isfile(checkpoint_path):
            logging.info("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])

        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logging.error(param)
                logging.error(name)
                logging.error("NaN in ", name)

        batch_instance = self.__load_stuff('batch_instance')
        batch_info = self.__load_stuff('batch_info')

        self._get_loss(batch_instance, batch_info)

    def test(self, test_in, model_dir, eval_dir):
        logging.info("Test on [%s] with model at [%s]" % (test_in, model_dir))

        evaluator = ImplicitEval(eval_dir)
        doc_count = 0

        best_model_path = os.path.join(self.model_dir, self.best_model_name)
        logging.info("Loading best model from '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        for test_data in self.reader.read_test_docs(
                data_gen(test_in), self.nid_detector):
            doc_id, instances, common_data, gold_labels, debug_data = test_data

            coh = self.model(instances, common_data)

            event_idxes = common_data['event_indices'].data.cpu().numpy()[
                0].tolist()
            slot_idxes = common_data['slot_indices'].data.cpu().numpy()[
                0].tolist()
            coh_scores = coh.data.cpu().numpy()[0].tolist()

            for (event_idxes, slot_idxes), result in groupby(
                    zip(zip(event_idxes, slot_idxes),
                        zip(coh_scores, gold_labels)), key=itemgetter(0)):
                score_labels = [r[1] for r in result]
                slot_name = self.reader.slot_names[int(slot_idxes)]
                evaluator.add_result(
                    doc_id, event_idxes, slot_name, score_labels, debug_data
                )

            doc_count += 1

            if doc_count % 10 == 0:
                logging.info("Processed %d documents." % doc_count)

        logging.info("Processed %d documents." % doc_count)

        logging.info("Writing evaluation output to %s." % eval_dir)
        evaluator.run()

    def train(self, train_in, validation_size=None, validation_in=None,
              model_out_dir=None, resume=False, track_pred=None):
        if track_pred is None:
            track_pred = {}

        target_pred_count = Counter()

        logging.info("Training with data from [%s]", train_in)

        if validation_in:
            logging.info("Validation with data from [%s]", validation_in)
        elif validation_size:
            logging.info(
                "Will use first [%d] lines for validation." % validation_size)
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

        previous_dev_loss = math.inf
        worse = 0

        log_freq = 100

        for epoch in range(start_epoch, self.nb_epochs):
            logging.info("Starting epoch {}.".format(epoch))

            for batch_data, debug_data in self.reader.read_cloze_batch(
                    data_gen(train_in), from_line=validation_size):

                batch_instance, batch_info, n_instance = batch_data

                for batch_preds in debug_data['predicate']:
                    for pred in batch_preds:
                        pred_text = pred.replace('-pred', '')

                        target_pred_count['_overall_'] += 1
                        if pred_text in track_pred:
                            target_pred_count[pred_text] += 1

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
                            logging.error(name, weight)

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

                    recent_loss = 0

            logging.info("Conducting validation.")

            dev_generator = None

            if validation_in:
                dev_generator = self.reader.read_cloze_batch(
                    data_gen(validation_in))

            if validation_size:
                dev_generator = self.reader.read_cloze_batch(
                    data_gen(train_in), until_line=validation_size)

            dev_loss, num_dev_batches, num_instances = self.validation(
                dev_generator)

            is_best = False
            if not best_loss or dev_loss < best_loss:
                best_loss = dev_loss
                is_best = True

            logging.info(
                "Finished epoch {epoch:d}, avg. loss {loss:.4f}, "
                "validation loss {dev_loss:.4f}{best:s}".format(
                    epoch=epoch, loss=total_loss / batch_count,
                    dev_loss=dev_loss / num_dev_batches,
                    best=', is current best.' if is_best else '.'
                ))

            for pred, count in target_pred_count.items():
                logging.info(
                    "Epoch %d: %s has been observed %d times." % (
                        epoch, pred, count)
                )

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
                        (
                            "Dev loss increase from {pre:.4f} to {curr:.4f}, "
                            "stop at Epoch {epoch:d}"
                        ).format(pre=previous_dev_loss, curr=dev_loss,
                                 epoch=epoch)
                    )
                    break

        for pred, count in target_pred_count.items():
            logging.info(
                "Overall, %s has been observed %d times." % (pred, count))


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
        log_dir = Unicode(help='logging directory').tag(config=True)
        do_training = Bool(help='Flag for conducting training.',
                           default_value=True).tag(config=True)
        do_test = Bool(help='Flag for conducting testing.',
                       default_value=False).tag(config=True)
        cmd_log = Bool(help='Log on command prompt only.',
                       default_value=False).tag(config=True)


    from event.util import load_config_with_cmd, load_with_sub_config
    import json

    conf = load_with_sub_config(sys.argv)

    basic_para = Basic(config=conf)

    from event.util import set_file_log, set_basic_log, ensure_dir
    from time import gmtime, strftime

    timestamp = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    if not basic_para.cmd_log and basic_para.log_dir:
        mode = ''

        if basic_para.do_training:
            mode += '_train'
        if basic_para.do_test:
            mode += '_test'

        log_path = os.path.join(
            basic_para.log_dir,
            basic_para.model_name + '_' + timestamp + mode + '.log')
        ensure_dir(log_path)
        set_file_log(log_path)
        print("Logging is set at: " + log_path)

    set_basic_log()

    logging.info("Started the runner at " + timestamp)
    logging.info(json.dumps(conf, indent=2))

    runner = ArgRunner(
        config=conf,
        model_dir=os.path.join(basic_para.model_dir, basic_para.model_name),
        debug_dir=basic_para.debug_dir
    )

    if basic_para.debug_dir and not os.path.exists(basic_para.debug_dir):
        os.makedirs(basic_para.debug_dir)

    target_predicates = {
        'bid', 'sell', 'loan', 'cost', 'plan', 'price', 'invest', 'price',
        'lose', 'invest', 'fund',
    }

    if basic_para.do_training:
        runner.train(
            basic_para.train_in,
            validation_size=basic_para.validation_size,
            validation_in=basic_para.valid_in,
            resume=True,
            track_pred=target_predicates,
        )

    if basic_para.do_test:
        eval_res_dir = os.path.join(
            basic_para.log_dir, basic_para.model_name + '_results'
        )

        if not os.path.exists(eval_res_dir):
            os.makedirs(eval_res_dir)

        print("Evaluation result will be saved in: " + eval_res_dir)

        runner.test(
            test_in=basic_para.test_in,
            model_dir=basic_para.model_dir,
            eval_dir=eval_res_dir,
        )
