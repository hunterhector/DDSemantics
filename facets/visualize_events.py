from forte import Pipeline
from forte.data.readers import DirPackReader
from forte.processors.stave import StaveProcessor

Pipeline(
    ontology_file="conf/full.json"
).set_reader(
    DirPackReader()
).add(
    StaveProcessor(),
    config={
        "port": 8880,
        "use_pack_name": True,
    }
).run(
    # "/home/hector/data/kbp/train"
    "/Users/hector.liu/Downloads/train"
)

