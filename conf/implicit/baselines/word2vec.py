# c.ArgModelPara.w2v_baseline_method = 'max_sim'  # max_sim, topk_average, average
# c.ArgModelPara.w2v_event_repr = 'concat'  # concat, sum
c.ArgModelPara.w2v_baseline_avg_topk = 3  # only when topk_average
c.ArgModelPara.model_type = ''

c.Basic.model_name = 'w2v_baseline'  # w2v_baseline, most_freq_baseline

c.Basic.run_baselines = True
c.Basic.do_training = False
c.Basic.do_test = False
