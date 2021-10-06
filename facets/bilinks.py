"""
Find bidirectional links between wiki pages.
"""
import pickle
import timeit
import datetime

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import DataPack
from forte.datasets.wikipedia.dbpedia.db_utils import ContextGroupedNIFReader, \
    get_resource_name, get_resource_fragment, print_progress
from forte.processors.base import PackProcessor
from ft.onto.wikipedia import WikiAnchor

def load_from_nif(link_file, output_file):
    linkings = {}
    bilinks = []

    num_articles = 0
    num_bilinks = 0

    start_time = timeit.default_timer()
    with open(output_file, "w") as out:
        for _, statements in ContextGroupedNIFReader(link_file):
            num_articles += 1

            for nif_range, rel, info in statements:
                r = get_resource_fragment(rel)
                if r is not None and r == "taIdentRef":
                    src_name = get_resource_name(nif_range)
                    target_name = get_resource_name(info)

                    if src_name == target_name:
                        continue

                    if linkings.get(target_name, None) == src_name:
                        bilinks.append((src_name, target_name))
                        linkings.pop(target_name)
                        num_bilinks += 1
                        out.write(f"{src_name}\t{target_name}\n")
                        out.flush()
                    else:
                        linkings[src_name] = target_name

            elapsed = timeit.default_timer() - start_time
            print_progress(
                f"{num_bilinks} bi-links found in {num_articles} after "
                f"{datetime.timedelta(seconds=elapsed)}, speed is "
                f"{num_articles / elapsed:.2f} (packs/second)."
            )


def main():
    import sys
    link_file = sys.argv[1]
    output_file = sys.argv[2]
    load_from_nif(link_file, output_file)


if __name__ == '__main__':
    main()
