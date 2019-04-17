import os

from traitlets import (
    Unicode,
    Bool,
    List,
)
from traitlets.config import Configurable

from event.io.dataset.ace import ACE
from event.io.dataset.conll import Conll
from event.io.dataset.framenet import FrameNet
from event.io.dataset.nombank import NomBank
from event.io.dataset.richere import RichERE
from event.io.dataset.propbank import PropBank
from event.io.dataset.negra import NeGraXML
from event.util import ensure_dir
from event.io.dataset.base import Corpus


def main(data_formats, config_files, output_config_file):
    from event.util import basic_console_log
    from event.util import load_file_config, load_config_with_cmd

    class OutputConf(Configurable):
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Text output directory').tag(config=True)
        brat_dir = Unicode(help='Brat visualization directory').tag(config=True)

    class EreConf(Configurable):
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

    class FrameNetConf(Configurable):
        fn_path = Unicode(help='FrameNet dataset path.').tag(config=True)

    class ConllConf(Configurable):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)

    class AceConf(Configurable):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Raw Text Output directory').tag(config=True)

    class NomBankConfig(Configurable):
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

    class PropBankConfig(Configurable):
        root = Unicode(help='Propbank corpus.').tag(config=True)
        propfile = Unicode(help='Prop File.').tag(config=True)
        frame_files = Unicode(help='Frame file pattern.').tag(config=True)
        verbs_file = Unicode(help='Verbs.').tag(config=True)

    class NegraConfig(Configurable):
        data_files = List(help='Input data path.', trait=Unicode).tag(
            config=True)

    basic_console_log()

    basic_params = []
    parsers = []

    output_param = OutputConf(config=load_file_config(output_config_file))

    # Create paths for output.
    if not os.path.exists(output_param.out_dir):
        os.makedirs(output_param.out_dir)

    if not os.path.exists(output_param.text_dir):
        os.makedirs(output_param.text_dir)

    brat_data_path = os.path.join(output_param.brat_dir, 'data')
    if not os.path.exists(brat_data_path):
        os.makedirs(brat_data_path)

    corpus = Corpus()

    for index, (data_format, config_file) in enumerate(
            zip(data_formats, config_files)):
        config = load_file_config(config_file)
        with_doc = index == 0
        if data_format == 'rich_ere':
            basic_param = EreConf(config=config)
            parser = RichERE(basic_param, corpus, with_doc)
        elif data_format == 'framenet':
            basic_param = FrameNetConf(config=config)
            parser = FrameNet(basic_param, corpus, with_doc)
        elif data_format == 'conll':
            basic_param = ConllConf(config=config)
            parser = Conll(basic_param, corpus, with_doc)
        elif data_format == 'ace':
            basic_param = AceConf(config=config)
            parser = ACE(basic_param, corpus, with_doc)
        elif data_format == 'nombank':
            basic_param = NomBankConfig(config=config)
            parser = NomBank(basic_param, corpus, with_doc)
        elif data_format == 'propbank':
            basic_param = PropBankConfig(config=config)
            parser = PropBank(basic_param, corpus, with_doc)
        elif data_format == 'negra':
            basic_param = NegraConfig(config=config)
            parser = NeGraXML(basic_param, corpus, with_doc)
        else:
            raise NotImplementedError("Selected format unknown.")

        basic_params.append(basic_param)
        parsers.append(parser)

    # Use the documents created by the first parser.
    for doc in parsers[0].get_doc():
        for basic_param, parser in zip(basic_params[1:], parsers[1:]):
            # TODO: This is not complete.
            # Add annotations from each parser.
            parser.add_all_annotations(doc)

        out_path = os.path.join(output_param.out_dir, doc.docid + '.json')
        ensure_dir(out_path)
        with open(out_path, 'w') as out:
            out.write(doc.dump(indent=2))

        out_path = os.path.join(output_param.text_dir, doc.docid + '.txt')
        ensure_dir(out_path)
        with open(out_path, 'w') as out:
            out.write(doc.doc_text)

        source_text, ann_text = doc.to_brat()
        out_path = os.path.join(output_param.brat_dir, 'data',
                                doc.docid + '.ann')
        ensure_dir(out_path)
        with open(out_path, 'w') as out:
            out.write(ann_text)

        out_path = os.path.join(output_param.brat_dir, 'data',
                                doc.docid + '.txt')
        ensure_dir(out_path)
        with open(out_path, 'w') as out:
            out.write(source_text)

    for p in parsers:
        p.print_stats()

    # Write brat configs.
    out_path = os.path.join(output_param.brat_dir, 'annotation.conf')
    with open(out_path, 'w') as out:
        out.write(corpus.get_brat_config())


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if len(args) > 0 and len(args) % 2 == 1:
        formats = args[0:-1:2]
        configs = args[1:-1:2]
        main(formats, configs, args[-1])
    else:
        raise ValueError("Argument incorrect.")
