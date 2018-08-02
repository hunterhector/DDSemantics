import os

# Model parameters
c.ModelPara.event_arg_vocab_size = 387354
c.ModelPara.event_embedding_dim = 300
c.ModelPara.word_vocab_size = 228575
c.ModelPara.word_embedding_dim = 300
c.ModelPara.arg_composition_layer_sizes = 600, 300
c.ModelPara.event_composition_layer_sizes = 400, 200
c.ModelPara.nb_epochs = 20
c.ModelPara.num_event_components = 8
c.ModelPara.num_extracted_features = 11
c.ModelPara.multi_context = True
c.ModelPara.max_events = 200
c.ModelPara.batch_size = 128
# Model parameters that changes the architectures
c.ModelPara.loss = 'cross_entropy'
c.ModelPara.vote_method = 'cosine'
c.ModelPara.vote_pooling = 'kernel'
# c.ModelPara.encode_distance = 'gaussian'
c.ModelPara.num_distance_features = 9

# Resources
# c.Resources.base = '/home/zhengzhl/workspace/implicit/gigaword_corpus/'
if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']
c.Resources.event_embedding_path = os.path.join(
    base, 'embeddings/event_frame_embeddings.pickle.wv.vectors.npy')
c.Resources.word_embedding_path = os.path.join(
    base, 'embeddings/word_embeddings.pickle.wv.vectors.npy')
c.Resources.event_vocab_path = os.path.join(
    base, 'embeddings/event_frame_embeddings.voc')
c.Resources.word_vocab_path = os.path.join(
    base, 'embeddings/word_embeddings.voc')
c.Resources.raw_lookup_path = os.path.join(base, 'vocab/')
# Runner parameters
c.Basic.train_in = os.path.join(base, 'hashed/partaa.gz')
# c.Basic.train_until = 1946546
c.Basic.validation_size = 1
c.Basic.debug_dir = os.path.join(base, 'debug')
c.Basic.model_name = 'basic'
c.Basic.model_dir = os.path.join(base, 'models', c.Basic.model_name)
