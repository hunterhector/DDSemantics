import os

if 'data_path' not in os.environ:
    raise KeyError("Please supply the main working directory as environment "
                   "variable: 'data_path'")
else:
    working = os.environ['data_path']

c.OutputConf.out_dir = os.path.join(working, 'nombank_with_gc', 'annotation')
c.OutputConf.text_dir = os.path.join(working, 'nombank_with_gc', 'text')
c.OutputConf.brat_dir = os.path.join(working, 'nombank_with_gc', 'brat')
