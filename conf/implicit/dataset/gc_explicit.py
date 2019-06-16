import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the main working directory as environment "
                   "variable: 'implicit_corpus'")
else:
    working = os.environ['implicit_corpus']

if 'eval_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'eval_corpus'")
else:
    corpus_path = os.environ['eval_corpus']

wsj_path = os.path.join(corpus_path, 'penn-treebank-rel3/parsed/mrg/wsj')
wsj_file_pattern = '\d\d/wsj_.*\.mrg'
base_out = os.path.join(working, 'nombank_explicit')

c.PropBankConfig.root = os.path.join(corpus_path, 'propbank-LDC2004T14')
c.PropBankConfig.propfile = os.path.join(corpus_path, 'propbank-LDC2004T14',
                                         'data', 'prop.txt')
c.PropBankConfig.frame_files = 'frames/.*\.xml'
c.PropBankConfig.verbs_file = os.path.join(corpus_path, 'data', 'verbs.txt')
c.PropBankConfig.wsj_path = wsj_path
c.PropBankConfig.wsj_file_pattern = wsj_file_pattern

c.OutputConf.out_dir = os.path.join(base_out, 'annotation')
c.OutputConf.text_dir = os.path.join(base_out, 'text')
c.OutputConf.brat_dir = os.path.join(base_out, 'brat')

c.PropBankConfig.order = 1
c.NomBankConfig.order = 0

c.NomBankConfig.nombank_path = os.path.join(corpus_path, 'nombank.1.0')
# The sorted version group data of the same doc together.
c.NomBankConfig.nomfile = os.path.join(corpus_path, 'nombank.1.0/nombank.1.0')
c.NomBankConfig.frame_file_pattern = 'frames/.*\.xml'
c.NomBankConfig.nombank_nouns_file = 'nombank.1.0.words'

c.NomBankConfig.wsj_path = wsj_path
c.NomBankConfig.wsj_file_pattern = wsj_file_pattern

c.NomBankConfig.implicit_path = os.path.join(
    corpus_path, 'GerberChai_annotations/implicit_argument_annotations.xml')

c.NomBankConfig.gc_only = True
c.NomBankConfig.explicit_only = True
c.NomBankConfig.stat_dir = os.path.join(base_out, 'stats')
