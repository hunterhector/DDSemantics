import os

if 'eval_corpus' not in os.environ:
    raise KeyError("Please supply the external data directory as environment "
                   "variable: 'eval_corpus'")
else:
    corpus_path = os.environ['eval_corpus']

c.PropBankConfig.root = os.path.join(corpus_path, 'propbank-LDC2004T14')
c.PropBankConfig.propfile = os.path.join(corpus_path, 'propbank-LDC2004T14',
                                         'data', 'prop.txt')
c.PropBankConfig.frame_files = 'frames/.*\.xml'
c.PropBankConfig.verbs_file = os.path.join(corpus_path, 'data', 'verbs.txt')

