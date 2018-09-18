from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Integer
)
import logging
import sys
from event.cross_ontology.srl_data_reader import SrlDataReader
from event.cross_ontology.srl_resources import SrlResources
from event.cross_ontology.srl_model_para import SrlModelPara


class SrlRunner(Configurable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.para = SrlModelPara(**kwargs)
        self.resources = SrlResources(**kwargs)

    def train(self, train_in):
        reader = SrlDataReader(self.resources.vocab)

        logging.info("Starting training from: {}".format(train_in))

        for tokens, tags in reader.read_data(train_in):
            print(tokens)
            print(tags)


if __name__ == '__main__':
    class Basic(Configurable):
        train_in = Unicode(help='training data directory').tag(config=True)
        model_name = Unicode(help='model name', default_value='basic').tag(
            config=True)
        model_dir = Unicode(help='model directory').tag(config=True)


    from event.util import basic_console_log
    from event.util import load_config_with_cmd

    basic_console_log()
    conf = load_config_with_cmd(sys.argv)

    logging.info("Started the SRL runner.")

    basic_para = Basic(config=conf)

    runner = SrlRunner(
        config=conf,
        model_dir=basic_para.model_dir,
    )

    runner.train(
        basic_para.train_in,
    )
