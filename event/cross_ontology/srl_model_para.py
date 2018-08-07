from traitlets.config import Configurable
from traitlets import (
    Int,
    Bool,
)


class SrlModelPara(Configurable):
    use_gpu = Bool(help='Whether to use gpu.', default_value=True).tag(
        config=True)
    word_vocab_size = Int(help='Vocabulary size of words').tag(config=True)
    word_embedding_dim = Int(
        help='Dimension of word embedding', default_value=300
    ).tag(config=True)
