import os

# Model parameters
c.ArgModelPara.model_type = 'EventPairComposition'
c.ArgModelPara.event_arg_vocab_size = 58983  # event_frame_embeddings_min500
c.ArgModelPara.event_embedding_dim = 300
c.ArgModelPara.word_vocab_size = 228575
c.ArgModelPara.word_embedding_dim = 300
c.ArgModelPara.arg_composition_layer_sizes = 600, 300
c.ArgModelPara.event_composition_layer_sizes = 400, 200
c.ArgModelPara.nb_epochs = 20
c.ArgModelPara.num_slots = 3
c.ArgModelPara.num_event_components = 8
c.ArgModelPara.num_extracted_features = 11
c.ArgModelPara.multi_context = True
c.ArgModelPara.max_events = 200
c.ArgModelPara.batch_size = 100
# Model parameters that changes the architectures
c.ArgModelPara.loss = 'cross_entropy'
c.ArgModelPara.vote_method = 'cosine'
c.ArgModelPara.vote_pooling = 'kernel'
# c.ArgModelPara.encode_distance = 'gaussian'
c.ArgModelPara.num_distance_features = 9

# How to detect Null Instantiation.
c.ArgModelPara.nid_method = 'gold'

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.ImplicitArgResources.event_embedding_path = os.path.join(
    base, 'gigaword_corpus',
    'embeddings/event_frame_embeddings_min500.pickle.wv.vectors.npy')
c.ImplicitArgResources.event_vocab_path = os.path.join(
    base, 'gigaword_corpus', 'embeddings/event_frame_embeddings_min500.voc')

c.ImplicitArgResources.word_embedding_path = os.path.join(
    base, 'gigaword_corpus', 'embeddings/word_embeddings.pickle.wv.vectors.npy')
c.ImplicitArgResources.word_vocab_path = os.path.join(
    base, 'gigaword_corpus', 'embeddings/word_embeddings.voc')
c.ImplicitArgResources.raw_lookup_path = os.path.join(base, 'gigaword_corpus',
                                                      'vocab/')
# Runner parameters
c.Basic.train_in = os.path.join(base, 'gigaword_corpus', 'hashed')
c.Basic.test_in = os.path.join(base, 'nombank_with_gc', 'processed',
                               'cloze_hashed_filter.json.gz')
c.Basic.validation_size = 10000
c.Basic.debug_dir = os.path.join(base, 'gigaword_corpus', 'debug')
c.Basic.log_dir = os.path.join(base, 'gigaword_corpus', 'logs')
c.Basic.model_dir = os.path.join(base, 'gigaword_corpus', 'models')

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
c.Basic.do_training = True
c.Basic.do_test = True
