import os

if "eval_corpus" not in os.environ:
    raise KeyError(
        "Please supply the external data directory as environment "
        "variable: 'eval_corpus'"
    )
else:
    corpus_path = os.environ["eval_corpus"]

if "implicit_corpus" not in os.environ:
    raise KeyError(
        "Please supply the external data directory as environment "
        "variable: 'implicit_corpus'"
    )
else:
    working = os.environ["implicit_corpus"]

path = os.path.join(corpus_path, "SemEval2010Task10", "test_gold")

c.NegraConfig.data_files = [os.path.join(path, p) for p in os.listdir(path)]

test_base = os.path.join(working, "semeval2010t10_test")

c.OutputConf.out_dir = os.path.join(test_base, "annotation")
c.OutputConf.text_dir = os.path.join(test_base, "text")
c.OutputConf.brat_dir = os.path.join(test_base, "brat")

c.NegraConfig.stat_out = os.path.join(test_base, "stats")
