import os

# Model parameters
c.ArgModelPara.event_arg_vocab_size = 387354
c.ArgModelPara.event_embedding_dim = 300
c.ArgModelPara.word_vocab_size = 228575
c.ArgModelPara.word_embedding_dim = 300
c.ArgModelPara.arg_composition_layer_sizes = 600, 300
c.ArgModelPara.event_composition_layer_sizes = 400, 200
c.ArgModelPara.nb_epochs = 20
c.ArgModelPara.num_event_components = 8
c.ArgModelPara.num_extracted_features = 11
c.ArgModelPara.multi_context = True
c.ArgModelPara.max_events = 200
c.ArgModelPara.batch_size = 128
# Model parameters that changes the architectures
c.ArgModelPara.loss = 'cross_entropy'
c.ArgModelPara.vote_method = 'cosine'
c.ArgModelPara.vote_pooling = 'kernel'
# c.ArgModelPara.encode_distance = 'gaussian'
c.ArgModelPara.num_distance_features = 9

# Null Instantiation.
c.ArgModelPara.nid_method = 'gold'

# ImplicitArgResources
# c.ImplicitArgResources.base = '/home/zhengzhl/workspace/implicit/gigaword_corpus/'

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']
c.ImplicitArgResources.event_embedding_path = os.path.join(
    base, 'embeddings/event_frame_embeddings.pickle.wv.vectors.npy')
c.ImplicitArgResources.word_embedding_path = os.path.join(
    base, 'embeddings/word_embeddings.pickle.wv.vectors.npy')
c.ImplicitArgResources.event_vocab_path = os.path.join(
    base, 'embeddings/event_frame_embeddings.voc')
c.ImplicitArgResources.word_vocab_path = os.path.join(
    base, 'embeddings/word_embeddings.voc')
c.ImplicitArgResources.raw_lookup_path = os.path.join(base, 'vocab/')
# Runner parameters
c.Basic.train_in = os.path.join(base, 'hashed')
# TODO: need to pin down the format quick.
c.Basic.test_in = os.path.join(base, 'hashed', 'partat.gz')
# c.Basic.train_until = 1946546
c.Basic.validation_size = 10000
c.Basic.debug_dir = os.path.join(base, 'debug')
c.Basic.log_dir = os.path.join(base, 'logs')
c.Basic.model_dir = os.path.join(base, 'models')

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
