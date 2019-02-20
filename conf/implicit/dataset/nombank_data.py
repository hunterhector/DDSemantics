import os

if 'eval_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'eval_corpus'")
else:
    corpus_path = os.environ['eval_corpus']

c.NomBankConfig.nombank_path = os.path.join(corpus_path, 'nombank.1.0')
# The sorted version group data of the same doc together.
c.NomBankConfig.nomfile = os.path.join(corpus_path, 'nombank.1.0/nombank.1.0')
c.NomBankConfig.frame_file_pattern = 'frames/.*\.xml'
c.NomBankConfig.nombank_nouns_file = 'nombank.1.0.words'

c.NomBankConfig.wsj_path = os.path.join(
    corpus_path, 'penn-treebank-rel3/parsed/mrg/wsj')
c.NomBankConfig.wsj_file_pattern = '\d\d/wsj_.*\.mrg'

c.NomBankConfig.implicit_path = os.path.join(
    corpus_path, 'GerberChai_annotations/implicit_argument_annotations.xml')

c.NomBankConfig.gc_only = True
