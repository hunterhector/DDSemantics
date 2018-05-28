from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
    Bool,
)


class ModelPara(Configurable):
    # Basic configs.
    use_gpu = Bool(help='Whether to use gpu.', default_value=True).tag(
        config=True)

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

    # Training controls
    early_stop_patience = Int(
        help='Early stop patience', default_value=1).tag(config=True)
    nb_epochs = Int(help='Number of epochs').tag(config=True)
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

    arg_composition_layer_sizes = List(
        Int, default_value=[600, 300],
        help='Output size of the argument composition layers.'
    ).tag(config=True)
    event_composition_layer_sizes = List(
        Int, default_value=[400, 200],
        help='Output size of the event composition layers.'
    ).tag(config=True)
    num_event_components = Int(
        help='Number of components per event').tag(config=True)
    num_extracted_features = Int(help='Feature size').tag(config=True)

    vote_method = Unicode(
        help='Method to pool compute the votes between two events',
        default_value='cosine'
    ).tag(config=True)

    vote_pooling = Unicode(
        help='Method to pool the multiple votes.',
        default_value='kernel'
    ).tag(config=True)

    coherence_method = Unicode(
        help='Method to compute the coherence between two events',
        default_value='attentive'
    ).tag(config=True)
