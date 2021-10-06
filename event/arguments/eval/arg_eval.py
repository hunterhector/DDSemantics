"""
Evaluate the Gerber & Chai corpus.
"""
from traitlets.config import Configurable
from traitlets import Unicode, Integer
from event.util import load_file_config
from nltk.corpus import (
    NombankCorpusReader,
    BracketParseCorpusReader,
)
from nltk.data import FileSystemPathPointer
import logging
import json


class ArgumentEvaluator:
    def __init__(self):
        pass


class ArgDataSet:
    def __init__(self, params):
        self.params = params

    def next_arg(self):
        pass


class GCDataSet(ArgDataSet):
    class GCConfig(Configurable):
        nombank_path = Unicode(help="Nombank corpus.").tag(config=True)
        nomfile = Unicode(help="Nombank file.").tag(config=True)
        frame_file_pattern = Unicode(help="Frame file pattern.").tag(config=True)
        nombank_nouns_file = Unicode(help="Nomank nous.").tag(config=True)

        # PennTree Bank config.
        wsj_path = Unicode(help="PennTree Bank path.").tag(config=True)
        wsj_file_pattern = Unicode(help="File pattern to read PTD data").tag(
            config=True
        )

    def __init__(self, config_path):
        conf = load_file_config(config_path)
        logging.info(json.dumps(conf, indent=2))

        params = GCDataSet.GCConfig(config=conf)
        super().__init__(params)

        wsj_treebank = BracketParseCorpusReader(
            root=params.wsj_path,
            fileids=params.wsj_file_pattern,
            tagset="wsj",
            encoding="ascii",
        )

        self.nombank = NombankCorpusReader(
            root=FileSystemPathPointer(params.nombank_path),
            nomfile=params.nomfile,
            framefiles=params.frame_file_pattern,
            nounsfile=params.nombank_nouns_file,
            parse_fileid_xform=lambda s: s[4:],
            parse_corpus=wsj_treebank,
        )

    def next_arg(self):
        for instance in self.nombank.instances():
            yield instance


class OntonoteDataSet(ArgDataSet):
    def __init__(self, config):
        super().__init__(config)


class SemEval10DataSet(ArgDataSet):
    def __init__(self, config):
        super().__init__(config)


if __name__ == "__main__":
    gc_dataset = GCDataSet("conf/implicit/nombank_data.py")
    for arg in gc_dataset.next_arg():
        print(arg)
        input("arg!")
