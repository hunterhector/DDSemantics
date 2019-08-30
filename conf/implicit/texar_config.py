import texar.torch as tx

hidden_dim = 512

arg_transformer = {
    "dim": hidden_dim,
    "num_blocks": 6,
    "multihead_attention": {
        "num_heads": 8,
        "output_dim": hidden_dim
        # See documentation for more optional hyperparameters
    },
    "initializer": {
        "type": "variance_scaling_initializer",
        "kwargs": {"factor": 1.0, "mode": "FAN_AVG", "uniform": True},
    },
    "poswise_feedforward": tx.modules.default_transformer_poswise_net_hparams(
        input_dim=hidden_dim,
        output_dim=hidden_dim
    ),
}
