import os

# Model parameters
c.ArgModelPara.encode_distance = "gaussian"

c.Basic.model_name = os.path.basename(__file__).replace(".py", "")
