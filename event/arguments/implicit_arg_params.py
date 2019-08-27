from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
    Bool,
)


class ArgModelPara(Configurable):
    # Basic configs.
    use_gpu = Bool(help='Whether to use gpu.', default_value=True).tag(
        config=True)
    model_type = Unicode(help='Type of the model.', default_value='').tag(
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
        help='Maximum number of events allowed per document during training',
        default_value=200).tag(config=True)
    max_cloze = Int(
        help='Maximum number of cloze to extract per document',
        default_value=150).tag(config=True)

    ## Model architecture related parameters.
    loss = Unicode(
        help='Loss type for implicit argument training',
        default_value='cross_entropy'
    ).tag(config=True)

    # The way to represent an event
    # Choices:
    # 1. fix_slots, when we have only predefined slots for an event
    # 2. role_dynamic, there are unknown number of slots, but each slot is
    # associated with a slot type.
    arg_representation_method = Unicode(
        help='The method used to represent the argument slots.').tag(
        config=True)

    # If we use role_dynamic, then how is the role combined to the arg?
    # Choices:
    # 1. biaffine
    # 2. dot product
    # 3. add
    # 4. concat
    arg_role_combine_func = 'biaffine'
    # role_compose_attention_method = 'biaffine' : renamed


    arg_composition_layer_sizes = List(
        Int, default_value=[600, 300],
        help='Output size of the argument composition layers.'
    ).tag(config=True)
    event_composition_layer_sizes = List(
        Int, default_value=[400, 200],
        help='Output size of the event composition layers.'
    ).tag(config=True)
    num_slots = Int(help='Number of slots in the model.').tag(config=True)

    # num_event_components = Int(
    #     help='Number of components per event').tag(config=True)
    use_frame = Bool(help='Whether to use frame in the model.').tag(config=True)
    slot_frame_formalism = Unicode(
        help='Which frame formalism is to predict the slots, currently support '
             'FrameNet and Propbank', default_value='Propbank'
    ).tag(config=True)
    num_extracted_features = Int(help='Feature size').tag(config=True)

    encode_distance = Unicode(
        help='Method to encode distances').tag(config=True)
    num_distance_features = Int(help='Distance features size').tag(config=True)

    vote_method = Unicode(
        help='Method to pool compute the votes between two events',
        default_value='cosine'
    ).tag(config=True)

    vote_pooling = Unicode(
        help='Method to pool the multiple votes.',
        default_value='kernel'
    ).tag(config=True)

    pool_topk = Int(
        help='The K value if Top K pooling is enabled.',
        default_value=3
    ).tag(config=True)

    # Null Instantiation Detector.
    nid_method = Unicode(
        help='The method for Null Instantiation Detector',
        default_value='logistic'
    ).tag(config=True)
    use_ghost = Bool(
        help='Use ghost instance as decision boundary', default_value=False
    ).tag(config=True)

    # Baseline parameters.
    w2v_baseline_method = Unicode(help='Baseline method type.',
                                  default_value='').tag(config=True)
    w2v_baseline_avg_topk = Int(help='Average top K', default_value=3).tag(
        config=True)
    w2v_event_repr = Unicode(help='Methods for creating the embedding.',
                             default_value='').tag(config=True)

    gold_field_name = Unicode(help='Field name for the gold standard').tag(
        config=True)
    factor_role = Unicode(
        help='The field name of the role that is used to modify the '
             'argument string as factors').tag(config=True)
