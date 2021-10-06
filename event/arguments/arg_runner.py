import pdb
import logging
import math
import os
import pickle
import shutil
from collections import Counter
import json
from time import localtime, strftime
import random
from typing import Dict

from smart_open import open
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Integer,
    Bool,
)
import torch
from torch.nn import functional as F
import numpy as np

from event.arguments.data.cloze_readers import HashedClozeReader
from event.arguments.NIFDetector import (
    GoldNullArgDetector,
    TrainableNullArgDetector,
    ResolvableArgDetector,
)
from event.arguments.implicit_arg_params import ArgModelPara
from event.arguments.arg_models import (
    EventCoherenceModel,
    BaselineEmbeddingModel,
    MostFrequentModel,
    RandomBaseline,
)
from event.arguments.implicit_arg_resources import ImplicitArgResources
from event.arguments.evaluation import ImplicitEval
from event.arguments.data.cloze_gen import ClozeSampler
from event.util import load_mixed_configs
from event.util import set_file_log, set_basic_log, ensure_dir, append_num_to_path

logger = logging.getLogger(__name__)


def data_gen(data_path, from_line=None, until_line=None):
    line_num = 0

    if os.path.isdir(data_path):
        last_file = None
        for f in sorted(os.listdir(data_path)):
            if not f.startswith(".") and f.endswith(".gz"):
                with open(os.path.join(data_path, f)) as fin:
                    for line in fin:
                        line_num += 1
                        if from_line and line_num <= from_line:
                            continue
                        if until_line and line_num > until_line:
                            break

                        if not last_file == f:
                            logger.info("Reading from {}".format(f))
                            last_file = f
                        yield line
    else:
        with open(data_path) as fin:
            logger.info("Reading from {}".format(data_path))
            for line in fin:
                line_num += 1
                if from_line and line_num <= from_line:
                    continue
                if until_line and line_num > until_line:
                    break
                yield line


def to_device(data, device):
    if isinstance(data, Dict):
        for key, d in data.items():
            data[key] = d.to(device)
        return data
    else:
        return data.to(device)


class CachableDataSource:
    def __init__(
        self, reader, data_iter, train_sampler, device, cache_size=-1, dump_dir=None
    ):
        self.data_source = data_iter
        self.train_sampler = train_sampler
        self.dump_dir = dump_dir
        self.reader = reader
        self.cache_size = cache_size
        self.device = device

    def check_dump_dir(self):
        return (
            self.dump_dir is not None
            and os.path.exists(self.dump_dir)
            and os.path.exists(os.path.join(self.dump_dir, ".success"))
        )

    def get_dump_files(self):
        file_list = [f for f in os.listdir(self.dump_dir) if f.endswith(".cache")]
        random.shuffle(file_list)
        return file_list

    def data(self):
        if self.check_dump_dir():
            logger.info("Reading from dumped instances.")
            for dump_f in self.get_dump_files():
                with open(os.path.join(self.dump_dir, dump_f), "rb") as f:
                    try:
                        while True:
                            yield to_device(pickle.load(f), self.device)
                    except EOFError:
                        pass
                    except Exception as e:
                        print(e)
                        pdb.set_trace()
        else:
            logger.info("Reading from raw data source.")
            train_gen = self.reader.read_train_batch(
                self.data_source, self.train_sampler
            )

            if self.dump_dir is not None and not os.path.exists(self.dump_dir):
                os.makedirs(self.dump_dir)

            # TODO: All data are cached here without sampling, but we can add
            #  sampling if we don't sample inside the reader, we can do them
            #  here?
            if self.dump_dir is not None:
                cache_path = os.path.join(self.dump_dir, f"dump_{0}.cache")
                cache_file = open(cache_path, "wb")
                cache_pair = 0, cache_file

            instance_idx = 0
            for data_batch in train_gen:
                yield [to_device(d, self.device) for d in data_batch]
                instance_idx += 1

                if self.dump_dir is not None:
                    cache_idx = int(instance_idx / self.cache_size)
                    if cache_idx == cache_pair[0]:
                        cache_file = cache_pair[1]
                    else:
                        cache_file = open(
                            os.path.join(self.dump_dir, f"dump_{cache_idx}.cache"), "wb"
                        )
                        cache_pair = cache_idx, cache_file

                    pickle.dump(data_batch, cache_file)

            if self.dump_dir is not None:
                with open(os.path.join(self.dump_dir, ".success"), "w") as f:
                    f.write("success")


class ArgRunner(Configurable):
    def __init__(self, **kwargs):
        super(ArgRunner, self).__init__(**kwargs)

        self.basic_para = Basic(**kwargs)
        self.para = ArgModelPara(**kwargs)
        self.resources = ImplicitArgResources(**kwargs)

        # Important, reader should be initialized earlier, because reader config
        # may change the embedding size (adding some extra words)
        self.reader = HashedClozeReader(self.resources, self.para)

        self.test_factor_role = self.basic_para.test_factor_role

        model_suffix = self.basic_para.model_name
        self.model_dir = os.path.join(self.basic_para.model_dir, model_suffix)
        self.debug_dir = os.path.join(self.basic_para.debug_dir, model_suffix)
        self.train_cache_dir = os.path.join(
            self.basic_para.train_cache_dir, self.basic_para.model_name
        )

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        logger.info("Model saving directory: " + self.model_dir)

        self.checkpoint_name = "checkpoint.pth"
        self.best_model_name = "model_best.pth"

        if self.para.use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                raise RuntimeError("CUDA not available.")
        else:
            self.device = "cpu"

        self.nb_epochs = self.para.nb_epochs

        if self.para.model_type:
            if self.para.model_type == "EventPairComposition":
                self.model = EventCoherenceModel(
                    self.para, self.resources, self.device, self.basic_para.model_name
                ).to(self.device)
                logger.info("Initialize model")
                logger.info(str(self.model))

        # Set up Null Instantiation detector.
        if self.para.nid_method == "gold":
            self.nid_detector = GoldNullArgDetector()
        elif self.para.nid_method == "train":
            self.nid_detector = TrainableNullArgDetector()

        self.resolvable_detector = ResolvableArgDetector()

    def _assert(self):
        if self.resources.word_embedding:
            assert self.para.word_vocab_size == self.resources.word_embedding.shape[0]
        if self.resources.event_embedding:
            assert (
                self.para.event_arg_vocab_size
                == self.resources.event_embedding.shape[0]
            )

    def _get_loss(self, labels, batch_instance, batch_common, mask):
        coh = self.model(batch_instance, batch_common)
        loss = F.binary_cross_entropy(coh * mask, labels)
        return loss

    def __dump_stuff(self, key, obj):
        logger.info("Saving object: {}.".format(key))
        with open(os.path.join(self.debug_dir, key + ".pickle"), "wb") as out:
            pickle.dump(obj, out)

    def __load_stuff(self, key):
        logger.info("Loading object: {}.".format(key))
        with open(os.path.join(self.debug_dir, key + ".pickle"), "rb") as fin:
            return pickle.load(fin)

    def __save_checkpoint(self, state, name):
        check_path = os.path.join(self.model_dir, name)
        logger.info("Saving model at {}".format(check_path))
        torch.save(state, check_path)
        return check_path

    @torch.no_grad()
    def validation(self, all_dev_data, dev_sampler):
        dev_loss = 0
        num_batches = 0
        num_instances = 0

        logger.info(f"Validation with {len(all_dev_data)} batches.")
        dev_sampler.reset()

        for dev_data in all_dev_data:
            labels, instances, common_data, b_size, mask, metadata = dev_data
            loss = self._get_loss(labels, instances, common_data, mask)

            if not loss:
                raise ValueError("Error in computing loss.")
            dev_loss += loss.item()

            num_batches += 1
            num_instances += b_size

        logger.info(
            "Validation loss is [%.4f] on [%d] batches, [%d] "
            "instances. Average loss is [%.4f]."
            % (dev_loss, num_batches, num_instances, dev_loss / num_batches)
        )

        return dev_loss, num_batches, num_instances

    def debug(self):
        checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])

        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logging.error(param)
                logging.error(name)
                logging.error("NaN in ", name)

    def __load_best(self):
        best_model_path = os.path.join(self.model_dir, self.best_model_name)
        if os.path.exists(best_model_path):
            logger.info("Loading best model from '{}'".format(best_model_path))
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.warning("Serialized model not existing, test without loading.")

    @torch.no_grad()
    def __test(self, model, test_lines, nid_detector, auto_test=False, eval_dir=None):
        self.model.eval()

        evaluator = ImplicitEval(eval_dir)
        instance_count = 0

        self.reader.auto_test = auto_test
        self.reader.factor_role = self.test_factor_role
        self.reader.use_gold_mention = True
        self.reader.use_auto_mention = False

        logger.info(
            f"During testing, factor role is [{self.reader.factor_role}], "
            f"use gold mention: {self.reader.use_gold_mention}, "
            f"use auto mention: {self.reader.use_auto_mention}"
        )

        logger.info(f"Evaluation result will be stored at {eval_dir}")

        for test_data in self.reader.read_test_docs(test_lines, nid_detector):
            (labels, instances, common_data, _, _, metadata) = test_data

            coh = model(
                to_device(instances, self.device), to_device(common_data, self.device)
            )

            coh_scores = np.squeeze(coh.data.cpu().numpy()).tolist()

            evaluator.add_prediction(coh_scores, metadata)

            instance_count += 1

            if instance_count % 1000 == 0:
                logger.info("Tested %d instances." % instance_count)

        if len(instance_count) == 0:
            logger.warning("0 instances found, check data reader.")
            print("0 instances found, check data reader!")

        logger.info("Finish testing %d instances." % instance_count)

        if eval_dir:
            logger.info("Writing evaluation output to %s." % eval_dir)

        evaluator.collect()

        self.model.train()

    def self_study_baseline(self, basic_para):
        dev_lines = [
            l
            for l in data_gen(basic_para.train_in, until_line=basic_para.self_test_size)
        ]

        # Random baseline.
        logger.info("Run self study with random baseline.")
        random_baseline = RandomBaseline(self.para, self.resources, self.device).to(
            self.device
        )
        self.__test(
            random_baseline,
            dev_lines,
            nid_detector=self.resolvable_detector,
            eval_dir=os.path.join(
                basic_para.log_dir,
                random_baseline.name,
                f"self_study_{basic_para.self_test_size}",
            ),
            test_factor_role=basic_para.test_factor_role,
            auto_test=True,
        )

        # W2v baseline.
        logger.info("Run self study with w2v baseline.")
        w2v_baseline = BaselineEmbeddingModel(
            self.para, self.resources, self.device
        ).to(self.device)
        self.__test(
            w2v_baseline,
            test_lines=dev_lines,
            nid_detector=self.resolvable_detector,
            eval_dir=os.path.join(
                basic_para.log_dir,
                w2v_baseline.name,
                f"self_study_{basic_para.self_test_size}",
            ),
            auto_test=True,
        )

        # Frequency baseline.
        logger.info("Run self study with frequency baseline.")
        most_freq_baseline = MostFrequentModel(
            self.para, self.resources, self.device
        ).to(self.device)
        self.__test(
            most_freq_baseline,
            test_lines=dev_lines,
            nid_detector=self.resolvable_detector,
            eval_dir=os.path.join(
                basic_para.log_dir,
                most_freq_baseline.name,
                f"self_study_{basic_para.self_test_size}",
            ),
            auto_test=True,
        )

    def self_study_model(self, basic_para, suffix, load_checkpoint=False):
        if load_checkpoint:
            # Checkpoint test.
            logger.info(f"Run self study with the checkpoint at {self.model_dir}.")

            checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
            if os.path.isfile(checkpoint_path):
                logger.info("Loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.info("Run self model with current parameters.")

        dev_lines = [
            l
            for l in data_gen(basic_para.train_in, until_line=basic_para.self_test_size)
        ]

        self.__test(
            self.model,
            test_lines=dev_lines,
            nid_detector=self.resolvable_detector,
            eval_dir=os.path.join(
                basic_para.log_dir,
                self.model.name,
                f"self_test_{basic_para.self_test_size}_{suffix}",
            ),
            auto_test=True,
        )
        logger.info("Done self test.")

    def run_baselines(self, basic_para):
        logger.info(f"Test baseline models on {basic_para.test_in}.")

        # W2v baseline.

        # # Variation 1: max sim, sum
        # self.para.w2v_baseline_method = 'max_sim'
        # self.para.w2v_event_repr = 'sum'
        # w2v_baseline = BaselineEmbeddingModel(
        #     self.para, self.resources, self.device).to(self.device)
        # self.__test(
        #     w2v_baseline, data_gen(basic_para.test_in),
        #     nid_detector=self.nid_detector,
        #     eval_dir=os.path.join(
        #         basic_para.log_dir, w2v_baseline.name, 'test', 'sum_max'
        #     ),
        # )
        #
        # # Variation 2: topk, sum
        # self.para.w2v_baseline_method = 'topk_average'
        # self.para.w2v_event_repr = 'sum'
        # w2v_baseline = BaselineEmbeddingModel(
        #     self.para, self.resources, self.device).to(self.device)
        # self.__test(
        #     w2v_baseline, data_gen(basic_para.test_in),
        #     nid_detector=self.nid_detector,
        #     eval_dir=os.path.join(
        #         basic_para.log_dir, w2v_baseline.name, 'test', 'sum_top3',
        #     ),
        # )

        # Frequency baseline.
        most_freq_baseline = MostFrequentModel(
            self.para, self.resources, self.device
        ).to(self.device)
        self.__test(
            most_freq_baseline,
            data_gen(basic_para.test_in),
            nid_detector=self.nid_detector,
            eval_dir=os.path.join(
                basic_para.log_dir,
                most_freq_baseline.name,
                "test",
                "default",
            ),
        )

        # Random baseline.
        random_baseline = RandomBaseline(self.para, self.resources, self.device).to(
            self.device
        )
        self.__test(
            random_baseline,
            data_gen(basic_para.test_in),
            nid_detector=self.nid_detector,
            eval_dir=os.path.join(
                basic_para.log_dir, random_baseline.name, "test", "default"
            ),
        )

    def test(self, test_in, eval_dir):
        logger.info("Test on [%s]." % test_in)
        self.__load_best()
        self.__test(self.model, data_gen(test_in), self.nid_detector, eval_dir=eval_dir)

    def train(self, basic_para, resume=False):
        train_in = basic_para.train_in
        target_pred_count = Counter()

        train_sampler = ClozeSampler()
        dev_sampler = ClozeSampler(seed=7)

        self.reader.factor_role = basic_para.train_factor_role
        self.reader.use_gold_mention = True
        self.reader.use_auto_mention = True
        logger.info(
            f"During training, factor role is [{self.reader.factor_role}], "
            f"use gold mention: {self.reader.use_gold_mention}, "
            f"use auto mention: {self.reader.use_auto_mention}"
        )

        logger.info("Training with data from [%s]", train_in)

        if self.basic_para.valid_in:
            logger.info("Validation with data from [%s]", self.basic_para.valid_in)
        elif self.basic_para.validation_size:
            logger.info(
                "Will use first [%d] lines for "
                "validation." % self.basic_para.validation_size
            )
        else:
            logging.error("No validation!")

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters())

        start_epoch = 0
        best_loss = math.inf
        previous_dev_loss = math.inf
        worse = 0

        if resume:
            checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
            if os.path.isfile(checkpoint_path):
                logger.info("Loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                start_epoch = checkpoint["epoch"]
                best_loss = checkpoint["best_loss"]
                previous_dev_loss = checkpoint["previous_dev_loss"]
                worse = checkpoint["worse"]

                # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
                del checkpoint
                torch.cuda.empty_cache()

                logger.info(
                    f"Loaded check point, epoch {start_epoch}, "
                    f"best loss {best_loss}, "
                    f"previous dev loss {previous_dev_loss}, "
                    f"worsen {worse} times."
                )
            else:
                logger.info(
                    "No model to resume at '{}', starting from scratch.".format(
                        checkpoint_path
                    )
                )

        # Read development lines.
        dev_lines = None
        if self.basic_para.valid_in:
            dev_lines = [l for l in data_gen(self.basic_para.valid_in)]
        if self.basic_para.validation_size:
            dev_lines = [
                l
                for l in data_gen(train_in, until_line=self.basic_para.validation_size)
            ]

        all_dev_data = []
        for dev_data in CachableDataSource(
            self.reader, dev_lines, dev_sampler, self.device
        ).data():
            all_dev_data.append(dev_data)

        if self.basic_para.pre_val:
            logger.info(
                "Conduct a pre-validation, this will overwrite best "
                "loss from the check point."
            )

            dev_loss, n_batches, n_instances = self.validation(
                all_dev_data, dev_sampler
            )

            best_loss = dev_loss
            previous_dev_loss = dev_loss

        batch_count = 0
        instance_count = 0

        # Training stats.
        total_loss = 0
        recent_loss = 0
        log_freq = 100

        train_dataset = CachableDataSource(
            self.reader,
            data_gen(train_in, from_line=self.basic_para.validation_size),
            train_sampler,
            self.device,
            self.basic_para.train_cache_size,
            self.train_cache_dir,
        )

        for epoch in range(start_epoch, self.nb_epochs):
            logger.info("Starting epoch {}.".format(epoch))
            epoch_batch_count = 0
            epoch_instance_count = 0

            train_sampler.reset()

            logger.info(
                f"Will ignore the first "
                f"{self.basic_para.validation_size} validation lines "
                f"for training."
            )

            for instance_data in train_dataset.data():
                labels, instances, batch_info, b_size, mask, _ = instance_data

                loss = self._get_loss(labels, instances, batch_info, mask)

                # Case of a bug.
                if not loss:
                    self.__save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "best_loss": best_loss,
                            "previous_dev_loss": previous_dev_loss,
                            "worse": worse,
                            "state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        "model_debug.pth",
                    )

                    for name, weight in self.model.named_parameters():
                        if name.startswith("event_to_var_layer"):
                            logging.error(name, weight)

                    self.__dump_stuff("batch_instance", instances)
                    self.__dump_stuff("batch_info", batch_info)

                    raise ValueError("Error in computing loss.")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TODO: found nan in the weights.
                nans_in_weight = torch.isnan(
                    self.model._linear_combine.weight
                ).nonzero()
                if nans_in_weight.shape[0] > 0:
                    pdb.set_trace()

                batch_count += 1
                epoch_batch_count += 1

                instance_count += b_size
                epoch_instance_count += b_size

                loss_val = loss.item()
                total_loss += loss_val
                recent_loss += loss_val

                if not batch_count % log_freq:
                    logger.info(
                        f"Epoch {epoch} ({epoch_batch_count} batches and "
                        f"{epoch_instance_count} instances); "
                        f"Total Batch {batch_count} ({instance_count} "
                        f"instances); Recent ({log_freq}) avg. "
                        f"loss {recent_loss / log_freq:.5f}; "
                        f"Overall avg. loss {total_loss / batch_count:.5f}"
                    )

                    if basic_para.self_test_size > 0:
                        self.self_study_model(basic_para, f"epoch_{epoch}")

                    recent_loss = 0

            logger.info("Computing validation loss.")
            dev_loss, n_batches, n_instances = self.validation(
                all_dev_data, dev_sampler
            )

            logger.info(
                f"Finished epoch {epoch:d}, "
                f"avg. training loss {total_loss / batch_count:.4f}, "
                f"validation loss {dev_loss / n_batches:.4f}"
            )

            new_best = False
            if not best_loss or dev_loss < best_loss:
                best_loss = dev_loss
                new_best = True

                logger.info(
                    f"Best loss is {best_loss:.4f} "
                    f"(avg. {best_loss / n_batches:.4f}). "
                    f"This is a new best."
                )
            else:
                logger.info(
                    f"Best loss is {best_loss:.4f} "
                    f"(avg. {best_loss / n_batches:.4f})."
                )

            if dev_loss < previous_dev_loss:
                previous_dev_loss = dev_loss
                worse = 0
            else:
                worse += 1

            checkpoint_path = self.__save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                    "previous_dev_loss": previous_dev_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "worse": worse,
                },
                self.checkpoint_name,
            )

            if new_best:
                best_path = os.path.join(self.model_dir, self.best_model_name)
                logger.info("Saving it as best model")
                shutil.copyfile(checkpoint_path, best_path)

            # Whether stop now.
            if worse == self.para.early_stop_patience:
                logger.info(
                    (
                        f"Dev loss increase from {previous_dev_loss:.4f} "
                        f"to {dev_loss:.4f}, stop at Epoch {epoch:d}"
                    )
                )
                break

        for pred, count in target_pred_count.items():
            logger.info("Overall, %s is observed %d times." % (pred, count))


def main(conf):
    basic_para = Basic(config=conf)

    if not basic_para.cmd_log and basic_para.log_dir:
        mode = ""

        if basic_para.do_training:
            mode += "_train"
        if basic_para.do_test:
            mode += "_test"

        log_path = os.path.join(
            basic_para.log_dir,
            basic_para.model_name,
            basic_para.run_name + mode + ".log",
        )

        append_num_to_path(log_path)
        ensure_dir(log_path)
        set_file_log(log_path)
        print("Logging is set at: " + log_path)

    if basic_para.debug_mode:
        set_basic_log(logging.DEBUG)
    else:
        set_basic_log()

    logger.info("Started the runner at " + strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    logger.info(json.dumps(conf, indent=2))

    runner = ArgRunner(config=conf)

    if basic_para.run_baselines:
        runner.run_baselines(basic_para)

    if basic_para.do_training:
        runner.train(basic_para, resume=True)

    if basic_para.do_test:
        if not basic_para.test_in:
            logger.error("Test data path is not provided.")

        if not os.path.exists(basic_para.test_in):
            logger.error(f"Cannot find test path at {basic_para.test_in}")

        result_dir = os.path.join(
            basic_para.log_dir,
            basic_para.model_name,
            basic_para.run_name,
            basic_para.test_data,
        )
        logger.info("Evaluation results will be saved in: " + result_dir)

        runner.test(
            test_in=basic_para.test_in,
            eval_dir=result_dir,
        )


if __name__ == "__main__":

    class Basic(Configurable):
        train_in = Unicode(help="Training data directory.").tag(config=True)
        test_data = Unicode(help="Test data name.").tag(config=True)
        test_in = Unicode(help="Testing data.").tag(config=True)
        test_out = Unicode(help="Test res.").tag(config=True)
        valid_in = Unicode(help="Validation in.").tag(config=True)
        validation_size = Integer(help="Validation size.").tag(config=True)
        self_test_size = Integer(help="Self test document size.", default_value=-1).tag(
            config=True
        )
        debug_dir = Unicode(help="Debug output.").tag(config=True)
        model_name = Unicode(help="Model name.", default_value="basic").tag(config=True)
        run_name = Unicode(help="Run name.", default_value="default").tag(config=True)
        train_cache_dir = Unicode(help="Path to training dump.").tag(config=True)
        train_cache_size = Integer(help="Number of items per cache").tag(config=True)
        model_dir = Unicode(help="Model directory.").tag(config=True)
        log_dir = Unicode(help="Logging directory.").tag(config=True)
        cmd_log = Bool(help="Log on command prompt only.", default_value=False).tag(
            config=True
        )
        pre_val = Bool(help="Pre-validate on the dev set.", default_value=False).tag(
            config=True
        )
        do_training = Bool(
            help="Flag for conducting training.", default_value=False
        ).tag(config=True)
        do_test = Bool(help="Flag for conducting testing.", default_value=False).tag(
            config=True
        )
        run_baselines = Bool(help="Run baseline.", default_value=False).tag(config=True)
        debug_mode = Bool(help="Debug mode", default_value=False).tag(config=True)

        test_factor_role = Unicode(
            help="The field name of the role that is used to "
            "determine the slot type."
        ).tag(config=True)
        train_factor_role = Unicode(
            help="The field name of the role that is used to "
            "determine the slot type."
        ).tag(config=True)

    conf = load_mixed_configs()
    main(conf)
