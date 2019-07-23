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
c.ArgModelPara.use_frame = True
# c.ArgModelPara.num_event_components = 8
c.ArgModelPara.num_extracted_features = 11
c.ArgModelPara.multi_context = True
c.ArgModelPara.max_events = 150
c.ArgModelPara.batch_size = 100
# Model parameters that changes the architectures
c.ArgModelPara.loss = 'cross_entropy'
c.ArgModelPara.vote_method = 'cosine'
c.ArgModelPara.vote_pooling = 'kernel'
# c.ArgModelPara.encode_distance = 'gaussian'
c.ArgModelPara.num_distance_features = 9
c.ArgModelPara.arg_representation_method = 'fix_slots'

# How to detect Null Instantiation.
c.ArgModelPara.nid_method = 'gold'
c.ArgModelPara.use_ghost = False

# Baseline stuff.
c.ArgModelPara.w2v_baseline_method = 'max_sim'  # max_sim, topk_average, average
c.ArgModelPara.w2v_event_repr = 'concat'  # concat, sum
c.ArgModelPara.w2v_baseline_avg_topk = 3  # only when topk_average

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.ImplicitArgResources.raw_corpus_name = 'gigaword_events'

c.ImplicitArgResources.event_embedding_path = os.path.join(
    base, c.ImplicitArgResources.raw_corpus_name,
    'embeddings/event_embeddings_mixed.pickle.wv.vectors.npy')
c.ImplicitArgResources.event_vocab_path = os.path.join(
    base, c.ImplicitArgResources.raw_corpus_name,
    'embeddings/event_embeddings_mixed.voc')
c.ImplicitArgResources.raw_lookup_path = os.path.join(
    base, c.ImplicitArgResources.raw_corpus_name, 'vocab/')

c.ImplicitArgResources.word_embedding_path = os.path.join(
    base, 'gigaword_word_embeddings', 'word_embeddings.pickle.wv.vectors.npy')
c.ImplicitArgResources.word_vocab_path = os.path.join(
    base, 'gigaword_word_embeddings', 'word_embeddings.voc')

# Runner parameters
c.Basic.train_in = os.path.join(base, c.ImplicitArgResources.raw_corpus_name,
                                'hashed')
c.Basic.validation_size = 10000
c.Basic.debug_dir = os.path.join(base, c.ImplicitArgResources.raw_corpus_name,
                                 'debug')
c.Basic.log_dir = os.path.join(base, c.ImplicitArgResources.raw_corpus_name,
                               'logs')
c.Basic.model_dir = os.path.join(base, c.ImplicitArgResources.raw_corpus_name,
                                 'models')

c.Basic.test_in = os.path.join(base, 'nombank_with_gc', 'processed',
                               'cloze_hashed.json.gz')

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
c.Basic.do_training = True
c.Basic.self_test_size = 100
c.Basic.do_test = True
c.Basic.gold_field_name = 'propbank_role'
