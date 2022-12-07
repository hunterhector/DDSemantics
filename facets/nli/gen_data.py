import sys

from facets.nli.nli_generator import NLIProcessor, TweakData
from facets.readers.nli_reader import MultiNLIReader
from facets.common.utils import ProgressPrinter
from forte import Pipeline
from forte.data.caster import MultiPackBoxer
from forte.data.selector import NameMatchSelector
from forte.processors.misc import RemoteProcessor
from forte.processors.writers import PackNameMultiPackWriter

output_dir = sys.argv[1]

Pipeline().set_reader(
    MultiNLIReader()
).add(
    # Call spacy on remote.
    RemoteProcessor(),
    config={
        "url": "http://localhost:8008"
    },
).add(
    # Call allennlp on remote.
    RemoteProcessor(),
    config={
        "url": "http://localhost:8009"
    },
).add(
    MultiPackBoxer()
).add(
    TweakData()
).add(
    NLIProcessor(),
    selector=NameMatchSelector(),
    selector_config={
        "select_name": "default",
        "reverse_selection": True,
    }
).add(
    PackNameMultiPackWriter(),
    config={
        "output_dir": output_dir
    }
).add(
    ProgressPrinter(),
).run()
