import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

# Model parameters
c.ArgModelPara.model_type = 'EventPairComposition'
c.ArgModelPara.use_frame = True
c.ArgModelPara.slot_frame_formalism = 'FrameNet'
c.ArgModelPara.arg_representation_method = 'role_dynamic'
c.ArgModelPara.arg_role_combine_func = 'mlp'
c.ArgModelPara.factor_role = 'fe'

c.Basic.test_in = os.path.join(base, 'semeval2010t10_train', 'processed',
                               'cloze_hashed.json.gz')

