from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
    Bool,
)


class ModelPara(Configurable):
    # Input size configs.
    event_arg_vocab_size = Int(
        help='Vocabulary size of events and argument words').tag(config=True)
    event_embedding_dim = Int(
        help='Dimension of event specific embedding', default_value=300
    ).tag(config=True)
    word_vocab_size = Int(help='Vocabulary size of words').tag(config=True)
    word_embedding_dim = Int(
        help='Dimension of word embedding', default_value=300
    ).tag(config=True)

    # Model layer sizes.
    arg_composition_layer_sizes = List(
        Int, default_value=[600, 300],
        help='Output size of the argument composition layers.'
    ).tag(config=True)
    event_composition_layer_sizes = List(
        Int, default_value=[400, 200],
        help='Output size of the event composition layers.'
    ).tag(config=True)
    nb_epochs = Int(help='Number of epochs').tag(config=True)
    num_event_components = Int(
        help='Number of components per event').tag(config=True)
    num_extracted_features = Int(help='Feature size').tag(config=True)
    batch_size = Int(help='Batch size', default_value=128).tag(config=True)

    multi_context = Bool(
        help='Whether to use only one context, '
             'or multiple context per document').tag(
        config=True)
    max_events = Int(
        help='Maximum number of events allowed per document',
        default_value=200).tag(config=True)

    # Model architecture related parameters.
    loss = Unicode(
        help='Loss type for implicit argument training',
        default_value='cross_entropy'
    ).tag(config=True)
