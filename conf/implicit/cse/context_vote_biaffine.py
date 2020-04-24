import os

# Model parameters
c.ArgModelPara.vote_method = 'biaffine'
c.ArgModelPara.early_stop_patience = 2

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
