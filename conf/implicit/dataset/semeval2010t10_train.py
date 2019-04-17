import os

if 'eval_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'eval_corpus'")
else:
    corpus_path = os.environ['eval_corpus']

c.NegraConfig.data_files = [os.path.join(
    corpus_path, 'SemEval2010Task10', 'train',
    'Semeval2010Task10TrainingFN', 'tiger',
    'TigerOfSanPedro.withHeads.xml')]
