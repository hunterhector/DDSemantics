from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
)


class ModelPara(Configurable):
    event_arg_vocab_size = Int(
        help='Vocabulary size of events and argument words').tag(config=True)
    event_embedding_dim = Int(
        help='Dimension of event specific embedding', default_value=300
    ).tag(config=True)
    word_vocab_size = Int(help='Vocabulary size of words').tag(config=True)
    word_embedding_dim = Int(
        help='Dimension of word embedding', default_value=300
    ).tag(config=True)
    arg_composition_layer_sizes = List(
        Int, default_value=[600, 300],
        help='Output size of the argument composition layers.'
    ).tag(config=True)
    event_composition_layer_sizes = List(
        Int, default_value=[400, 200],
        help='Output size of the event composition layers.'
    ).tag(config=True)

    nb_epochs = Int(help='Number of epochs').tag(config=True)
    num_args = Int(help='Number of args per event').tag(config=True)

    num_extracted_features = Int(help='Feature size').tag(config=True)

    batch_size = Int(help='Batch size').tag(config=True)
