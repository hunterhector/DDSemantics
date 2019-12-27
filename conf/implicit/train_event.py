import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

# No factor_role provided, so the system will use default dep_slot.
# c.ArgModelPara.factor_role = ''
c.ArgModelPara.use_auto_mention = True
c.ArgModelPara.use_gold_mention = False

raw_corpus_name = 'gigaword_events'

# Runner parameters
c.Basic.train_in = os.path.join(base, raw_corpus_name, 'hashed_test')
c.Basic.debug_dir = os.path.join(base, raw_corpus_name, 'debug')
c.Basic.train_dump = os.path.join(base, raw_corpus_name, 'train_cache')
c.Basic.train_cache_size = 1000
