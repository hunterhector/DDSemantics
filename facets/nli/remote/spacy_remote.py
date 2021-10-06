from forte import Pipeline
from forte.data.readers import RawDataDeserializeReader
from fortex.spacy import SpacyProcessor

Pipeline().set_reader(RawDataDeserializeReader()).add(
    SpacyProcessor(),
    config={
        "processors": ["sentence"]
    }
).serve(port=8008)
