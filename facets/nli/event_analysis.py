from facets.readers.kbp_reader import EREReader
from forte import Pipeline

import sys

kbp_dir = sys.argv[1]

Pipeline().set_reader(
    EREReader()
).run(
    [kbp_dir]
)
