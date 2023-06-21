import sys

from forte import Pipeline
from forte.data.readers import DirPackReader
from forte.processors.writers import PackNameJsonPackWriter

from facets.coref_decode import SameTokenCoref
from facets.writer import TbfWriter


def main(input_path: str):
    pipeline = Pipeline()

    pipeline.set_reader(
        DirPackReader(),
        config={
            "serialize_method": "jsonpickle"
        }
    ).add(
        TbfWriter(),
        config={
            "output_path": None,
            "system_name": None,
        }
    ).initialize()

    pipeline.process(
        input_path
    )


if __name__ == '__main__':
    main(sys.argv[1])
