import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.HashParam.event_vocab = os.path.join(
    base, 'gigaword_events', 'embeddings', 'event_embeddings_mixed.voc')
c.HashParam.word_vocab = os.path.join(
    base, 'gigaword_word_embeddings', 'word_embeddings.voc')
c.HashParam.frame_dep_map = os.path.join(
    base, 'frame_maps', 'frames_args_filled.tsv')
c.HashParam.dep_frame_map = os.path.join(
    base, 'frame_maps', 'args_frames_filled.tsv')
c.HashParam.prop_dep_map = os.path.join(
    base, 'frame_maps', 'prop_dep_all.tsv')
c.HashParam.component_vocab_dir = os.path.join(
    base, 'gigaword_events', 'vocab/')
c.HashParam.framenet_frame_files = os.path.join(
    base, 'resources', 'fndata-1.5', 'frame')
c.HashParam.nombank_frame_files = os.path.join(
    base, 'resources', 'nombank.1.0', 'frames')
c.HashParam.nom_map = 'resources/nombankArgMap.tsv'
c.HashParam.frame_formalism = 'Propbank'
