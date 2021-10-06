import os
import sys

from forte.data import DataPack
from forte.datasets.wikipedia.dbpedia.db_utils import print_progress

from onto.facets import WikiCategory

if __name__ == "__main__":
    base_dir = sys.argv[1]
    pack_dir = os.path.join(base_dir, "packs")
    file_suffix = ".json.gz"

    target_dir = os.path.join(pack_dir, "nif_raw_struct_links_bak")
    target_index = os.path.join(target_dir, "category.idx")
    true_index = os.path.join(target_dir, "article.idx")

    true_path_lookup = {}
    with open(true_index) as ti:
        for line in ti:
            article_name, article_path = line.strip().split("\t")
            true_path_lookup[article_name] = article_path

    print(f"Loaded {len(true_path_lookup)} true article paths.")

    with open(target_index) as ti:
        doc_count = 0
        removed_count = 0
        fixed_count = 0

        for line in ti:
            article_name, article_path = line.strip().split("\t")
            true_path = true_path_lookup[article_name]
            doc_count += 1

            import pdb

            if true_path == article_path:
                print_progress(
                    f"Fixing category from {true_path} ({doc_count}"
                    f"/{removed_count}/{fixed_count})"
                )
                fixed_count += 1
            else:
                print_progress(
                    f"Removing extra file {article_path}, will keep "
                    f"{true_path} ({doc_count}/{removed_count}/{fixed_count})"
                )
                removed_count += 1

                print(
                    f"Removing extra file {article_path}, will keep "
                    f"{true_path} ({doc_count}/{removed_count}/{fixed_count})"
                )
                pdb.set_trace()

