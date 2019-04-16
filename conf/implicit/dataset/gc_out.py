import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the main working directory as environment "
                   "variable: 'implicit_corpus'")
else:
    working = os.environ['implicit_corpus']

c.OutputConf.out_dir = os.path.join(working, 'nombank_with_gc', 'annotation')
c.OutputConf.text_dir = os.path.join(working, 'nombank_with_gc', 'text')
c.OutputConf.brat_dir = os.path.join(working, 'nombank_with_gc', 'brat')
