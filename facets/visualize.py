import sys

from forte import Pipeline
from forte.data.readers import DirPackReader
from forte.processors.stave import StaveProcessor

if __name__ == '__main__':
    input_dir = sys.argv[1]
    onto_file = sys.argv[2]
    nlp = Pipeline(ontology_file=onto_file)

    nlp.set_reader(
        DirPackReader(), config={
            "suffix": ".json.gz"
        }
    ).add(
        StaveProcessor()
    ).run(input_dir)
