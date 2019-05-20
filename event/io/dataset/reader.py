import os

from traitlets import (
    Unicode,
    Bool,
    List,
    Int
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


def main():
    from event.util import basic_console_log
    from event.util import load_file_config, load_mixed_configs

    class OutputConf(Configurable):
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Text output directory').tag(config=True)
        brat_dir = Unicode(help='Brat visualization directory').tag(config=True)

    class DataConf(Configurable):
        # Default is just a large number to rank it late.
        order = Int(help='Order of this parser', default_value=10000000).tag(
            config=True)

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
        format = Unicode(help='name for format', default_value='ERE').tag(
            config=False)

    class FrameNetConf(DataConf):
        fn_path = Unicode(help='FrameNet dataset path.').tag(config=True)
        format = Unicode(help='name for format', default_value='FrameNet').tag(
            config=True)

    class ConllConf(DataConf):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)
        format = Unicode(help='name for format', default_value='ConllConf').tag(
            config=True)

    class AceConf(DataConf):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Raw Text Output directory').tag(config=True)
        format = Unicode(help='name for format', default_value='ACE').tag(
            config=True)

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
        format = Unicode(help='name for format', default_value='NomBank').tag(
            config=True)
        stat_dir = Unicode(help='Path for stats.').tag(config=True)

    class PropBankConfig(DataConf):
        root = Unicode(help='Propbank corpus.').tag(config=True)
        propfile = Unicode(help='Prop File.').tag(config=True)
        frame_files = Unicode(help='Frame file pattern.').tag(config=True)
        verbs_file = Unicode(help='Verbs.').tag(config=True)
        format = Unicode(help='name for format', default_value='PropBank').tag(
            config=True)

        # PennTree Bank config.
        wsj_path = Unicode(help='PennTree Bank path.').tag(config=True)
        wsj_file_pattern = Unicode(help='File pattern to read PTD data').tag(
            config=True)

    class NegraConfig(DataConf):
        data_files = List(help='Input data path.', trait=Unicode).tag(
            config=True)
        stat_out = Unicode(help="Output statistics").tag(config=True)
        format = Unicode(help='name for format', default_value='Negra').tag(
            config=True)

    basic_console_log()

    basic_params = []
    order_parsers = []

    config = load_mixed_configs()

    output_param = OutputConf(config=config)

    # Create paths for output.
    if not os.path.exists(output_param.out_dir):
        os.makedirs(output_param.out_dir)

    if not os.path.exists(output_param.text_dir):
        os.makedirs(output_param.text_dir)

    brat_data_path = os.path.join(output_param.brat_dir, 'data')
    if not os.path.exists(brat_data_path):
        os.makedirs(brat_data_path)

    corpus = Corpus()
    order_parsers = []

    # with_doc = index == 0
    if 'RichERE' in config:
        basic_param = EreConf(config=config)
        o = basic_param.order
        parser = RichERE(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))
    if 'FrameNetConf' in config:
        basic_param = FrameNetConf(config=config)
        o = basic_param.order
        parser = FrameNet(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))
    if 'ConllConf' in config:
        basic_param = ConllConf(config=config)
        o = basic_param.order
        parser = Conll(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))
    if 'AceConf' in config:
        basic_param = AceConf(config=config)
        o = basic_param.order
        parser = ACE(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))
    if 'NomBankConfig' in config:
        basic_param = NomBankConfig(config=config)
        o = basic_param.order
        parser = NomBank(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))
        print('found nombank config')
    if 'PropBankConfig' in config:
        basic_param = PropBankConfig(config=config)
        o = basic_param.order
        parser = PropBank(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))
        print('found propbank config')
    if 'NegraConfig' in config:
        basic_param = NegraConfig(config=config)
        o = basic_param.order
        parser = NeGraXML(basic_param, corpus, o == 0)
        order_parsers.append((o, parser))

    order_parsers.sort()
    parsers = [p[1] for p in order_parsers]

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
    main()
