import os

if 'srl_workspace' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'srl_workspace'")
else:
    base = os.environ['srl_workspace']

c.SrlResources.word_embedding_path = '/home/zhengzhl/workspace/resources/embeddings/glove.6B/glove.test.txt'
c.SrlResources.embedding_type = 'glove'
c.SrlResources.word_vocab_path = os.path.join(base, 'word_vocab.pickle')

c.SrlModelPara.use_gpu = True
c.SrlModelPara.word_vocab_size = 400000
c.SrlModelPara.word_embedding_dim = 300

c.Basic.train_in = '/home/zhengzhl/workspace/datasets/srl/rich_ere'
c.Basic.model_name = 'sequential'
c.Basic.model_dir = os.path.join(base, c.Basic.model_name)
