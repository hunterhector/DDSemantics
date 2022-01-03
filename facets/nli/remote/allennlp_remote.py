from fortex.allennlp import AllenNLPProcessor

from forte import Pipeline
from forte.data.readers import RawDataDeserializeReader

Pipeline().set_reader(RawDataDeserializeReader()).add(
    AllenNLPProcessor(),
    config={
        "processors": ["tokenize", "pos", "lemma", "depparse", "srl"],
        "infer_batch_size": 1,
    }
).serve(port=8009)
