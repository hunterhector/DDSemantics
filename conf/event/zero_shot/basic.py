import os

if 'event_resources' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'event_resources'")
else:
    base = os.environ['event_resources']

c.ZeroShotEventResources.event_embedding_path = os.path.join(
    base, 'event_embeddings',
    'event_embeddings_mixed.pickle.wv.vectors.npy')
c.ZeroShotEventResources.event_vocab_path = os.path.join(
    base, 'event_embeddings','event_embeddings_mixed.voc')
c.ZeroShotEventResources.word_embedding_path = os.path.join(
    base, 'event_embeddings', 'word_embeddings.pickle.wv.vectors.npy')
c.ZeroShotEventResources.word_vocab_path = os.path.join(
    base, 'event_embeddings', 'word_embeddings.voc')
c.ZeroShotEventResources.target_ontology = 'resources/LDCOntology_v0.1.jsonld'

c.Basic.input_path = '/home/hector/workspace/aida/test_docs/rich/simple_run'
c.Basic.output_path = '/home/hector/workspace/aida/test_docs/rich/simple_run_mapped'