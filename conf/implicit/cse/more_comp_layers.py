import os

c.ArgModelPara.arg_composition_layer_sizes = 600, 300, 300
c.ArgModelPara.event_composition_layer_sizes = 400, 200, 200

c.Basic.model_name = os.path.basename(__file__).replace(".py", "")
