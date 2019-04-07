import os

c.ArgModelPara.baseline_method = 'max_sim'
c.ArgModelPara.baseline_avg_topk = 3
c.ArgModelPara.model_type = ''

c.Basic.model_name = 'max_sim_baseline'
c.Basic.run_baselines = True
c.Basic.do_training = False
c.Basic.do_test = False

