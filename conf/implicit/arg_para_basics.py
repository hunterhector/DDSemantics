import os

# Model parameters
c.ArgModelPara.model_type = 'EventPairComposition'
c.ArgModelPara.num_slots = 3
c.ArgModelPara.use_frame = True

c.ArgModelPara.event_arg_vocab_size = 58983  # event_frame_embeddings_min500
c.ArgModelPara.arg_composition_layer_sizes = 600, 300
c.ArgModelPara.event_composition_layer_sizes = 400, 200
c.ArgModelPara.event_embedding_dim = 300
c.ArgModelPara.word_vocab_size = 228575
c.ArgModelPara.word_embedding_dim = 300
c.ArgModelPara.nb_epochs = 20
c.ArgModelPara.num_extracted_features = 11
c.ArgModelPara.max_events = 120
c.ArgModelPara.batch_size = 512

# Model parameters that changes the architectures
c.ArgModelPara.multi_context = True
c.ArgModelPara.loss = 'cross_entropy'
c.ArgModelPara.vote_method = 'cosine'
c.ArgModelPara.vote_pooling = 'kernel'
# c.ArgModelPara.encode_distance = 'gaussian'
c.ArgModelPara.arg_role_combine_func = ''
c.ArgModelPara.num_distance_features = 9

# How to detect Null Instantiation.
c.ArgModelPara.nid_method = 'gold'
c.ArgModelPara.use_ghost = False

# Baseline stuff.
c.ArgModelPara.w2v_baseline_method = 'max_sim'  # max_sim, topk_average, average
c.ArgModelPara.w2v_event_repr = 'concat'  # concat, sum
c.ArgModelPara.w2v_baseline_avg_topk = 3  # only when topk_average

# Important slot names.
c.ArgModelPara.gold_role_field = 'gold_role_id'

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.ImplicitArgResources.word_embedding_path = os.path.join(
    base, 'gigaword_word_embeddings', 'word_embeddings.pickle.wv.vectors.npy')
c.ImplicitArgResources.word_vocab_path = os.path.join(
    base, 'gigaword_word_embeddings', 'word_embeddings.voc')

c.ImplicitArgResources.min_vocab_count = 50

c.Basic.validation_size = 10000

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
c.Basic.run_baselines = False
c.Basic.do_training = True
c.Basic.self_test_size = -1
c.Basic.do_test = True
