import os

c.ArgModelPara.baseline_method = 'max_sim'
c.ArgModelPara.baseline_avg_topk = 3

c.Basic.model_name = os.path.basename(__file__).replace('.py', '')
