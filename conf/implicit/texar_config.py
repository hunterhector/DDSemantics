import texar.torch as tx

hidden_dim = 50

arg_transformer = {
    "dim": hidden_dim,
    "num_blocks": 2,
    "multihead_attention": {
        "num_heads": 1,
        "num_units": hidden_dim,
        "output_dim": hidden_dim,
    },
    "initializer": {
        "type": "variance_scaling_initializer",
        "kwargs": {"factor": 1.0, "mode": "FAN_AVG", "uniform": True},
    },
    "poswise_feedforward": tx.modules.default_transformer_poswise_net_hparams(
        input_dim=hidden_dim, output_dim=hidden_dim
    ),
}
