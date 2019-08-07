import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

# Model parameters
c.ArgModelPara.model_type = 'EventPairComposition'
c.ArgModelPara.num_slots = 3
c.ArgModelPara.use_frame = True
c.ArgModelPara.slot_frame_formalism = 'Propbank'
c.ArgModelPara.arg_representation_method = 'fix_slots'

c.Basic.test_in = os.path.join(base, 'nombank_with_gc', 'processed',
                               'cloze_hashed.json.gz')

