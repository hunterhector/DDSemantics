import sys
from pprint import pprint as pp

import IPython

from forte import Pipeline
from forte.data import DataPack
from forte.data.readers.deserialize_reader import SinglePackReader
from forte.processors.base import PackProcessor

from facets.wiki.processors.wiki import WikiEntityCompletion


def pe(pack: DataPack, class_name: str):
    pp(entries(pack, class_name))


def entries(pack: DataPack, class_name: str):
    return [item for item in pack.get(class_name)]


class PackExplorer(PackProcessor):
    def _process(self, pack: DataPack):
        IPython.embed()


if __name__ == "__main__":
    Pipeline().set_reader(SinglePackReader()).add(WikiEntityCompletion()).add(
        PackExplorer()
    ).run(sys.argv[1])
