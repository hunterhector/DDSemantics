import sys

import IPython

from facets.utils import ProgressPrinter
from forte import Pipeline
from forte.data import DataPack
from forte.data.readers.deserialize_reader import DirPackReader
from forte.processors.base import PackProcessor


class PackExplorer(PackProcessor):
    def _process(self, pack: DataPack):
        IPython.embed()


if __name__ == "__main__":
    Pipeline().set_reader(
        DirPackReader(),
        config={
            "suffix": ".pickle.gz",
            "zip_pack": True,
            "serialize_method": "pickle"
        },
    ).add(
        ProgressPrinter()
    ).run(sys.argv[1])
