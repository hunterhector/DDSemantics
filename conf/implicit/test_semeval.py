import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.Basic.test_in = os.path.join(base, 'semeval2010t10_train', 'processed',
                               'cloze_hashed_filter.json.gz')
