import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

# Runner parameters
c.Basic.train_in = os.path.join(base, 'gigaword_corpus', 'hashed')
