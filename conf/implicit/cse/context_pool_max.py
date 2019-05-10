import os

# Model parameters
c.ArgModelPara.vote_pooling = 'max'

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
