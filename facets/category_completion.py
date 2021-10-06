""" Preprocess wikipedia data"""
import logging
import os
import pickle
import sys

from forte import Pipeline
from forte.common import Resources
from forte.datasets.wikipedia.dbpedia import WikiArticleWriter

from facets.wiki.processors.wiki import WikiCategoryReader

if __name__ == "__main__":
    base_dir = sys.argv[1]
    pack_dir = os.path.join(base_dir, "packs")

    redirect_map = pickle.load(
        open(os.path.join(pack_dir, "redirects.pickle"), "rb"))
    resources = Resources()
    resources.update(redirects=redirect_map)

    # Define paths
    pack_input = os.path.join(pack_dir, "nif_raw_struct_links")
    # Should write to same files.
    # pack_output = pack_input
    pack_output = os.path.join(pack_dir, "category")
    # Store which documents have category.
    pack_input_index = os.path.join(pack_input, "article.idx")
    # Store which documents have category.
    pack_output_index = os.path.join(pack_output, "category.idx")

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        filename=os.path.join(pack_dir, "category.log"),
    )

    Pipeline(resources).set_reader(
        WikiCategoryReader(),
        config={
            "pack_index": pack_input_index,
            "pack_dir": pack_input,
        },
    ).add(
        WikiArticleWriter(),
        config={
            "output_dir": pack_output,
            "zip_pack": True,
            "drop_record": True,
            "input_index_file": pack_input_index,
            "output_index_file": pack_output_index,
            "use_input_index": True,
            "overwrite": True,
        },
    ).run(os.path.join(base_dir, "article_categories_en.tql.bz2"))
