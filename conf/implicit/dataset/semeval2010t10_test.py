import os

if 'eval_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'eval_corpus'")
else:
    corpus_path = os.environ['eval_corpus']

path = os.path.join(corpus_path, 'SemEval2010Task10', 'test_gold')

c.NegraConfig.data_files = [os.path.join(path, p) for p in os.listdir(path)]
