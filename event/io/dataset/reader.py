import os

from traitlets import (
    Unicode,
    Bool,
)
from traitlets.config import Configurable

from event.io.dataset.ace import ACE
from event.io.dataset.conll import Conll
from event.io.dataset.framenet import FrameNet
from event.io.dataset.nombank import NomBank
from event.io.dataset.richere import RichERE
from event.util import ensure_dir


def main(data_format, args):
    from event.util import basic_console_log
    from event.util import load_file_config, load_config_with_cmd

    class DataConf(Configurable):
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Text output directory').tag(config=True)
        brat_dir = Unicode(help='Brat visualization directory').tag(config=True)

    class EreConf(DataConf):
        source = Unicode(help='Plain source input directory').tag(config=True)
        ere = Unicode(help='ERE input data').tag(config=True)
        src_ext = Unicode(help='Source file extension',
                          default_value='.xml').tag(config=True)
        ere_ext = Unicode(help='Ere file extension',
                          default_value='.rich_ere.xml').tag(config=True)
        ere_split = Bool(help='Whether split ere based on the file names').tag(
            config=True)
        ignore_quote = Bool(help='model name', default_value=False).tag(
            config=True)

    class FrameNetConf(DataConf):
        fn_path = Unicode(help='FrameNet dataset path.').tag(config=True)

    class ConllConf(DataConf):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)

    class AceConf(DataConf):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Raw Text Output directory').tag(config=True)

    class NomBankConfig(DataConf):
        nombank_path = Unicode(help='Nombank corpus.').tag(config=True)
        nomfile = Unicode(help='Nombank file.').tag(config=True)
        frame_file_pattern = Unicode(help='Frame file pattern.').tag(
            config=True)
        nombank_nouns_file = Unicode(help='Nomank nous.').tag(config=True)

        # PennTree Bank config.
        wsj_path = Unicode(help='PennTree Bank path.').tag(config=True)
        wsj_file_pattern = Unicode(help='File pattern to read PTD data').tag(
            config=True)

        implicit_path = Unicode(help='Implicit annotation xml path.').tag(
            config=True)
        gc_only = Bool(help='Only use GC arguments.').tag(config=True)

    basic_console_log()

    if os.path.exists(args[0]):
        config = load_file_config(args[0])
    else:
        config = load_config_with_cmd(args)

    if data_format == 'rich_ere':
        basic_para = EreConf(config=config)
        parser = RichERE(basic_para)
    elif data_format == 'framenet':
        basic_para = FrameNetConf(config=config)
        parser = FrameNet(basic_para)
    elif data_format == 'conll':
        basic_para = ConllConf(config=config)
        parser = Conll(basic_para)
    elif data_format == 'ace':
        basic_para = AceConf(config=config)
        parser = ACE(basic_para)
    elif data_format == 'nombank':
        basic_para = NomBankConfig(config=config)
        parser = NomBank(basic_para)
    else:
        basic_para = None
        parser = None

    if parser:
        if not os.path.exists(basic_para.out_dir):
            os.makedirs(basic_para.out_dir)

        if not os.path.exists(basic_para.text_dir):
            os.makedirs(basic_para.text_dir)

        brat_data_path = os.path.join(basic_para.brat_dir, 'data')
        if not os.path.exists(brat_data_path):
            os.makedirs(brat_data_path)

        for doc in parser.get_doc():
            out_path = os.path.join(basic_para.out_dir, doc.docid + '.json')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(doc.dump(indent=2))

            out_path = os.path.join(basic_para.text_dir, doc.docid + '.txt')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(doc.doc_text)

            source_text, ann_text = doc.to_brat()
            out_path = os.path.join(basic_para.brat_dir, 'data',
                                    doc.docid + '.ann')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(ann_text)

            out_path = os.path.join(basic_para.brat_dir, 'data',
                                    doc.docid + '.txt')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(source_text)

        out_path = os.path.join(basic_para.brat_dir, 'annotation.conf')
        with open(out_path, 'w') as out:
            out.write(parser.corpus.get_brat_config())


if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2:])
