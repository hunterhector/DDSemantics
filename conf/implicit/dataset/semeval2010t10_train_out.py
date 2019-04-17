import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'implicit_corpus'")
else:
    working = os.environ['implicit_corpus']

c.OutputConf.out_dir = os.path.join(working, 'semeval2010t10_train',
                                    'annotation')
c.OutputConf.text_dir = os.path.join(working, 'semeval2010t10_train', 'text')
c.OutputConf.brat_dir = os.path.join(working, 'semeval2010t10_train', 'brat')
