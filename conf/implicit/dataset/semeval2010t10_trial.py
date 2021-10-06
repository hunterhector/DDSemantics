import os

if "implicit_corpus" not in os.environ:
    raise KeyError(
        "Please supply the external data directory as environment "
        "variable: 'implicit_corpus'"
    )
else:
    working = os.environ["implicit_corpus"]

if "eval_corpus" not in os.environ:
    raise KeyError(
        "Please supply the external data directory as environment "
        "variable: 'eval_corpus'"
    )
else:
    corpus_path = os.environ["eval_corpus"]

c.NegraConfig.data_files = [
    os.path.join(corpus_path, "SemEval2010Task10", "task10-fn_trial.xml")
]

trial_base = os.path.join(working, "semeval2010t10_trial")

c.OutputConf.out_dir = os.path.join(trial_base, "annotation")
c.OutputConf.text_dir = os.path.join(trial_base, "text")
c.OutputConf.brat_dir = os.path.join(trial_base, "brat")

c.NegraConfig.stat_out = os.path.join(trial_base, "stats")
