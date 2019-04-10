from traitlets.config import Configurable
from event.arguments.impicit_arg_params import ArgModelPara
from event.arguments.arg_models import (
    EventPairCompositionModel,
    BaselineEmbeddingModel,
)
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
from event import torch_util
from event.arguments.util import ClozeSampler


def data_gen(data_path, from_line=None, until_line=None):
    line_num = 0

    if os.path.isdir(data_path):
        last_file = None
        for f in sorted(os.listdir(data_path)):
            with smart_open(os.path.join(data_path, f)) as fin:
                for line in fin:
                    line_num += 1
                    if from_line and line_num <= from_line:
                        continue
                    if until_line and line_num > until_line:
                        break

                    if not last_file == f:
                        logging.info("Reading from {}".format(f))
                        last_file = f
                    yield line
    else:
        with smart_open(data_path) as fin:
            logging.info("Reading from {}".format(data_path))
            for line in fin:
                line_num += 1
                if from_line and line_num <= from_line:
                    continue
                if until_line and line_num > until_line:
                    break
                yield line


class NullArgDetector:
    def __init__(self):
        pass

    def should_fill(self, event_info, slot):
        pass


class GoldNullArgDetector(NullArgDetector):
    """
    A Null arg detector that look at gold standard.
    """

    def __index__(self):
        super(NullArgDetector, self).__init__()

    def should_fill(self, event_info, slot):
        return event_info['args'][slot].get('implicit', False)


class AllArgDetector(NullArgDetector):
    """
    A Null arg detector that returns true for every arg.
    """

    def __init__(self):
        super(NullArgDetector, self).__init__()

    def should_fill(self, event_info, slot):
        return len(event_info['args'][slot]) > 0


class TrainableNullArgDetector(NullArgDetector):
    """
    A Null arg detector that is trained to predict.
    """

    def __index__(self):
        super(NullArgDetector, self).__init__()

    def should_fill(self, doc_info, arg_info):
        raise NotImplementedError


class ArgRunner(Configurable):
    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)

        self.para = ArgModelPara(**kwargs)
        self.resources = ImplicitArgResources(**kwargs)

        self.model_dir = self.para.model_dir
        self.debug_dir = self.para.debug_dir

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
        self.nb_epochs = self.para.nb_epochs

        if self.para.model_type:
            if self.para.model_type == 'EventPairComposition':
                self.model = EventPairCompositionModel(
                    self.para, self.resources).to(self.device)
                logging.info("Initialize model")
                logging.info(str(self.model))

        self.reader = HashedClozeReader(self.resources,
                                        self.para.batch_size,
                                        multi_context=self.para.multi_context,
                                        max_events=self.para.max_events,
                                        max_cloze=self.para.max_cloze)

        # Set up Null Instantiation detector.
        if self.para.nid_method == 'gold':
            self.nid_detector = GoldNullArgDetector()
        elif self.para.nid_method == 'train':
            self.nid_detector = TrainableNullArgDetector()

        self.all_detector = AllArgDetector()

    def _assert(self):
        if self.resources.word_embedding:
            assert self.para.word_vocab_size == \
                   self.resources.word_embedding.shape[0]
        if self.resources.event_embedding:
            assert self.para.event_arg_vocab_size == \
                   self.resources.event_embedding.shape[0]

    def __loss(self, l_label, l_scores, mask):
        if self.para.loss == 'cross_entropy':
            total_loss = 0

            for label, scores in zip(l_label, l_scores):
                masked_scores = torch.zeros(scores.shape).to(self.device)
                masked_scores.masked_scatter_(mask, scores)

                if label == 1:
                    # Make the coh scores close to 1 when these are actual
                    # answers.
                    gold = torch.zeros(scores.shape).to(
                        self.device).masked_fill_(mask, 1)
                else:
                    # Make the coh score close to 0 when these are fake cases.
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

    def _compute_coh(self, batch_instance, batch_common):
        correct_coh = self.model(batch_instance['gold'], batch_common)
        cross_coh = self.model(batch_instance['cross'], batch_common)
        inside_coh = self.model(batch_instance['inside'], batch_common)
        return correct_coh, cross_coh, inside_coh

    def _get_accuracy(self, cross_coh, inside_coh):
        pass

    def _get_loss(self, batch_instance, batch_common, mask):
        outputs = self._compute_coh(batch_instance, batch_common)
        labels = [1, 0, 0]
        loss = self.__loss(labels, outputs, mask)
        return loss

    def __dump_stuff(self, key, obj):
        logging.info("Saving object: {}.".format(key))
        with open(os.path.join(self.debug_dir, key + '.pickle'), 'wb') as out:
            pickle.dump(obj, out)

    def __load_stuff(self, key):
        logging.info("Loading object: {}.".format(key))
        with open(os.path.join(self.debug_dir, key + '.pickle'), 'rb') as fin:
            return pickle.load(fin)

    def __save_checkpoint(self, state, name):
        check_path = os.path.join(self.model_dir, name)
        logging.info("Saving model at {}".format(check_path))
        torch.save(state, check_path)
        return check_path

    def validation(self, dev_lines, dev_sampler):
        dev_loss = 0
        num_batches = 0
        num_instances = 0

        dev_sampler.reset()
        with torch.no_grad():
            for batch_data, debug_data in self.reader.read_train_batch(
                    dev_lines, dev_sampler):
                batch_instance, batch_common, b_size, mask = batch_data

                loss = self._get_loss(batch_instance, batch_common, mask)

                if not loss:
                    raise ValueError('Error in computing loss.')
                dev_loss += loss.item()

                num_batches += 1
                num_instances += b_size

            logging.info("Validation loss is [%.4f] on [%d] batches, [%d] "
                         "instances. Average loss is [%.4f]." % (
                             dev_loss, num_batches, num_instances,
                             dev_loss / num_batches))

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

    def __load_best(self):
        best_model_path = os.path.join(self.model_dir, self.best_model_name)
        logging.info("Loading best model from '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def __test(self, model, test_lines, nid_detector, eval_dir=None):
        evaluator = ImplicitEval(eval_dir)
        doc_count = 0

        for test_data in self.reader.read_test_docs(test_lines, nid_detector):
            doc_id, instances, common_data, gold_labels, debug_data = test_data

            coh = model(instances, common_data)

            event_idxes = common_data['event_indices'].data.cpu().numpy()[
                0].tolist()
            slot_idxes = common_data['slot_indices'].data.cpu().numpy()[
                0].tolist()
            coh_scores = coh.data.cpu().numpy()[0].tolist()

            for (event_idx, slot_idx), result in groupby(
                    zip(zip(event_idxes, slot_idxes),
                        zip(coh_scores, gold_labels),
                        debug_data['predicate'],
                        debug_data['entity_text'],
                        debug_data['gold_entity'],
                        ),
                    key=itemgetter(0)):

                _, score_labels, debug_preds, debug_entities, golds = zip(
                    *result)

                slot_name = self.reader.slot_names[slot_idx]

                evaluator.add_result(
                    doc_id, event_idx, slot_name, score_labels,
                    {
                        'predicate': debug_preds[0],
                        'entity_text': debug_entities,
                        'gold_entity': golds[0],
                    }
                )

            doc_count += 1

            if doc_count % 1000 == 0:
                logging.info("Tested %d documents." % doc_count)

        logging.info("Finish testing %d documents." % doc_count)

        if eval_dir:
            logging.info("Writing evaluation output to %s." % eval_dir)

        evaluator.run()

    def run_baseline(self, test_in, eval_dir):
        logging.info("Test on [%s]." % test_in)

        base_model = BaselineEmbeddingModel(self.para, self.resources).to(
            self.device)
        base_model.eval()
        self.__test(base_model, data_gen(test_in),
                    nid_detector=self.nid_detector, eval_dir=eval_dir)

    def test(self, test_in, eval_dir):
        logging.info("Test on [%s]." % test_in)
        self.__load_best()
        self.model.eval()
        self.__test(self.model, data_gen(test_in), self.nid_detector, eval_dir)

    def train(self, train_in, validation_size=None, validation_in=None,
              model_out_dir=None, resume=False, track_pred=None,
              pre_validation=False):
        if track_pred is None:
            track_pred = {}

        target_pred_count = Counter()

        train_sampler = ClozeSampler()
        dev_sampler = ClozeSampler(seed=7)

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

        batch_count = 0
        instance_count = 0

        start_epoch = 0
        best_loss = math.inf
        previous_dev_loss = math.inf
        worse = 0

        if resume:
            checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
            if os.path.isfile(checkpoint_path):
                logging.info("Loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                previous_dev_loss = checkpoint['previous_dev_loss']
                worse = checkpoint['worse']
            else:
                logging.info(
                    "No model to resume at '{}', starting from scratch.".format(
                        checkpoint_path)
                )

        # Read development lines.
        dev_lines = None
        if validation_in:
            dev_lines = [l for l in data_gen(validation_in)]
        if validation_size:
            dev_lines = [l for l in
                         data_gen(train_in, until_line=validation_size)]

        if pre_validation:
            logging.info("Conduct a pre-validation, this will overwrite best "
                         "loss with the most recent loss.")
            for i in range(5):
                dev_loss, n_batches, n_instances = self.validation(
                    dev_lines, dev_sampler)
                input('tried validation once')
            best_loss = dev_loss
            previous_dev_loss = dev_loss

        # Training stats.
        total_loss = 0
        recent_loss = 0
        log_freq = 100

        for epoch in range(start_epoch, self.nb_epochs):
            logging.info("Starting epoch {}.".format(epoch))

            train_sampler.reset()
            for batch_data, debug_data in self.reader.read_train_batch(
                    data_gen(train_in, from_line=validation_size),
                    train_sampler):

                batch_instance, batch_info, b_size, mask = batch_data

                for batch_preds in debug_data['predicate']:
                    for pred in batch_preds:
                        pred_text = pred.replace('-pred', '')
                        target_pred_count['_overall_'] += 1
                        if pred_text in track_pred:
                            target_pred_count[pred_text] += 1

                loss = self._get_loss(batch_instance, batch_info, mask)

                if not loss:
                    self.__save_checkpoint({
                        'epoch': epoch + 1,
                        'best_loss': best_loss,
                        'previous_dev_loss': previous_dev_loss,
                        'worse': worse,
                        'state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, 'model_debug.pth')

                    for name, weight in self.model.named_parameters():
                        if name.startswith('event_to_var_layer'):
                            logging.error(name, weight)

                    self.__dump_stuff('batch_instance', batch_instance)
                    self.__dump_stuff('batch_info', batch_info)

                    raise ValueError('Error in computing loss.')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1
                instance_count += b_size

                loss_val = loss.item()
                total_loss += loss_val
                recent_loss += loss_val

                if not batch_count % log_freq:
                    logging.info(
                        "Epoch {}, Batch {} ({} instances); Recent (100) avg. "
                        "loss {:.5f}; Overall avg. loss {:.5f}".format(
                            epoch, batch_count, instance_count,
                            recent_loss / log_freq,
                            total_loss / batch_count)
                    )
                    recent_loss = 0

            checkpoint_path = self.__save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'previous_dev_loss': previous_dev_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'worse': worse,
            }, self.checkpoint_name)

            logging.info("Computing validation loss.")
            dev_loss, n_batches, n_instances = self.validation(
                dev_lines, dev_sampler)

            logging.info("Computing test result on dev set.")
            self.model.eval()
            self.__test(self.model, test_lines=dev_lines,
                        nid_detector=self.all_detector)
            self.model.train()

            logging.info(
                "Finished epoch {epoch:d}, avg. loss {loss:.4f}, "
                "validation loss {dev_loss:.4f}".format(
                    epoch=epoch, loss=total_loss / batch_count,
                    dev_loss=dev_loss / n_batches,
                ))

            if not best_loss or dev_loss < best_loss:
                best_loss = dev_loss
                best_path = os.path.join(self.model_dir, self.best_model_name)
                logging.info("Saving it as best model")
                shutil.copyfile(checkpoint_path, best_path)

            logging.info("Best loss is %.4f." % best_loss)

            for pred, count in target_pred_count.items():
                logging.info(
                    "Epoch %d: %s is observed %d times." % (epoch, pred, count)
                )

            if dev_loss < previous_dev_loss:
                previous_dev_loss = dev_loss
                worse = 0
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
            logging.info("Overall, %s is observed %d times." % (pred, count))


if __name__ == '__main__':
    class Basic(Configurable):
        train_in = Unicode(help='Training data directory.').tag(config=True)
        test_in = Unicode(help='Testing data.').tag(config=True)
        test_out = Unicode(help='Test res.').tag(config=True)
        valid_in = Unicode(help='Validation in.').tag(config=True)
        validation_size = Integer(help='Validation size.').tag(config=True)
        debug_dir = Unicode(help='Debug output.').tag(config=True)
        model_name = Unicode(help='Model name.', default_value='basic').tag(
            config=True)
        run_name = Unicode(help='Run name.', default_value='default').tag(
            config=True)
        model_dir = Unicode(help='Model directory.').tag(config=True)
        log_dir = Unicode(help='Logging directory.').tag(config=True)
        cmd_log = Bool(help='Log on command prompt only.',
                       default_value=False).tag(config=True)
        pre_val = Bool(help='Pre-validate on the dev set.',
                       default_value=False).tag(config=True)
        do_training = Bool(help='Flag for conducting training.',
                           default_value=False).tag(config=True)
        do_test = Bool(help='Flag for conducting testing.',
                       default_value=False).tag(config=True)
        run_baselines = Bool(help='Run baseline.', default_value=False).tag(
            config=True)


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
            basic_para.model_name,
            basic_para.run_name + "_" + timestamp + mode + '.log')
        ensure_dir(log_path)
        set_file_log(log_path)
        print("Logging is set at: " + log_path)

    set_basic_log()

    logging.info("Started the runner at " + timestamp)
    logging.info(json.dumps(conf, indent=2))

    runner = ArgRunner(
        config=conf,
        model_dir=os.path.join(
            basic_para.model_dir, basic_para.model_name
        ),
        debug_dir=basic_para.debug_dir,
    )

    if basic_para.debug_dir and not os.path.exists(basic_para.debug_dir):
        os.makedirs(basic_para.debug_dir)

    target_predicates = {
        'bid', 'sell', 'loan', 'cost', 'plan', 'price', 'invest', 'price',
        'lose', 'invest', 'fund',
    }

    result_dir = os.path.join(
        basic_para.log_dir, basic_para.model_name, 'results',
        basic_para.run_name + "_" + timestamp,
    )

    if basic_para.run_baselines:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        print("Baseline evaluation results will be saved in: " + result_dir)

        runner.run_baseline(test_in=basic_para.test_in, eval_dir=result_dir)

    if basic_para.do_training:
        runner.train(
            basic_para.train_in,
            validation_size=basic_para.validation_size,
            validation_in=basic_para.valid_in,
            resume=True,
            track_pred=target_predicates,
            pre_validation=basic_para.pre_val,
        )

    if basic_para.do_test:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print("Evaluation results will be saved in: " + result_dir)

        runner.test(
            test_in=basic_para.test_in,
            eval_dir=result_dir,
        )
