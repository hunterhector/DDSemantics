import os

if 'eval_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'eval_corpus'")
else:
    base = os.environ['eval_corpus']

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the main working directory as environment "
                   "variable: 'implicit_corpus'")
else:
    working = os.environ['implicit_corpus']

c.NomBankConfig.out_dir = os.path.join(working, 'nombank_with_gc', 'annotation')
c.NomBankConfig.text_dir = os.path.join(working, 'nombank_with_gc', 'text')

c.NomBankConfig.nombank_path = os.path.join(base, 'nombank.1.0')
# The sorted version group data of the same doc together.
c.NomBankConfig.nomfile = os.path.join(base, 'nombank.1.0/nombank.1.0.sorted')
c.NomBankConfig.frame_file_pattern = 'frames/.*\.xml'
c.NomBankConfig.nombank_nouns_file = 'nombank.1.0.words'

c.NomBankConfig.wsj_path = os.path.join(
    base, 'penn-treebank-rel3/parsed/mrg/wsj')
c.NomBankConfig.wsj_file_pattern = '\d\d/wsj_.*\.mrg'

c.NomBankConfig.implicit_path = os.path.join(
    base, 'GerberChai_annotations/implicit_argument_annotations.xml')
