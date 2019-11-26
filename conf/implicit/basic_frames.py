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

raw_corpus_name = 'gigaword_frames'

c.ImplicitArgResources.event_embedding_path = os.path.join(
    base, raw_corpus_name,
    'embeddings/event_embeddings_mixed.pickle.wv.vectors.npy')
c.ImplicitArgResources.event_vocab_path = os.path.join(
    base, raw_corpus_name, 'embeddings/event_embeddings_mixed.voc')
c.ImplicitArgResources.raw_lookup_path = os.path.join(
    base, raw_corpus_name, 'vocab/')

# Runner parameters
c.Basic.log_dir = os.path.join(base, raw_corpus_name, 'logs')
c.Basic.model_dir = os.path.join(base, raw_corpus_name, 'models')
