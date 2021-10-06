from forte import Pipeline
from forte.data.readers import RawDataDeserializeReader, StringReader
from fortex.allennlp import AllenNLPProcessor
from fortex.nltk import NLTKSentenceSegmenter

# Pipeline().set_reader(
#     StringReader()
# ).add(
#     NLTKSentenceSegmenter()
# ).add(
#     AllenNLPProcessor(),
#     config={
#         "processors": ["tokenize,depparse,srl"],
#         # "srl_url": "https://storage.googleapis.com/allennlp-public-models"
#         #            "/structured-prediction-srl-bert.2020.12.15.tar.gz"
#     }
# ).run("This is a test. \n This is another test. \n")

Pipeline().set_reader(RawDataDeserializeReader()).add(
    AllenNLPProcessor(),
    config={
        "processors": ["tokenize,depparse,srl"],
        # "srl_url": "https://storage.googleapis.com/allennlp-public-models"
        #            "/structured-prediction-srl-bert.2020.12.15.tar.gz"
    }
).serve(port=8009)
