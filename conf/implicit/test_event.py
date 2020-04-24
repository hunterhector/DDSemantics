import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

# Runner parameters
c.Basic.test_in = os.path.join(base, 'nombank_with_gc', 'processed',
                               'cloze_hashed.json.gz')
c.Basic.do_test = True
c.Basic.run_baselines = False
c.Basic.test_factor_role = 'gold_role_id'
