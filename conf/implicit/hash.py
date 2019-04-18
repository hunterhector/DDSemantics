import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.HashParam.event_vocab = os.path.join(
    base, 'gigaword_corpus', 'embeddings/event_frame_embeddings.voc')
c.HashParam.word_vocab = os.path.join(
    base, 'gigaword_corpus', 'embeddings/word_embeddings.voc')
c.HashParam.frame_arg_map = os.path.join(
    base, 'gigaword_corpus', 'frame_maps/frames_args_filled.tsv')
c.HashParam.dep_frame_map = os.path.join(
    base, 'gigaword_corpus', 'frame_maps/args_frames_filled.tsv')
c.HashParam.component_vocab_dir = os.path.join(
    base, 'gigaword_corpus', 'vocab/')
