"""Pipeline to clean up the wikipedia pages."""

import logging
import os
import pickle
import sys

from forte import Pipeline
from forte.common import Resources
from forte.data.readers import DirPackReader
from forte.datasets.wikipedia.dbpedia import WikiArticleWriter
from forte.processors.nlp.subword_tokenizer import SubwordTokenizer
from fortex.spacy import SpacyProcessor
from smart_open import open

from facets.common.utils import ProgressPrinter
from facets.wiki.processors.wiki import WikiAddTitle


def complete_and_tokens():
    # Define paths
    pack_input = os.path.join(pack_dir, "nif_raw_struct_links")
    pack_output = os.path.join(pack_dir, "nif_raw_struct_links_token")
    # Store which documents are processed, try to make input output structure
    # similar.
    pack_input_index = os.path.join(pack_input, "article.idx")
    pack_output_index = os.path.join(pack_output, "article.idx")

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        filename=os.path.join(pack_dir, "complete_tokenize.log"),
    )

    pipeline = Pipeline(loaded_resource).set_reader(
        DirPackReader(),
        config={
            "suffix": ".json.gz",
            "zip_pack": True
        },
    ).add(
        WikiAddTitle()
    ).add(
        SpacyProcessor(),
        config={
            "processors": ["sentence", "tokenize"],
        }
    ).add(
        SubwordTokenizer(),
        config={
            "tokenizer_configs": {
                "pretrained_model_name": "bert-base-uncased"
            },
            "token_source": "ft.onto.base_ontology.Token",
        }
    ).add(
        WikiArticleWriter(),
        config={
            "output_dir": pack_output,
            "zip_pack": True,
            "drop_record": True,
            "input_index_file": pack_input_index,
            "output_index_file": pack_output_index,
            "use_input_index": True,
            "serialize_method": "jsonpickle"
        },
    ).add(ProgressPrinter())
    pipeline.run(pack_input)


if __name__ == "__main__":
    base_dir = sys.argv[1]
    pack_dir = os.path.join(base_dir, "packs")

    redirect_map = pickle.load(
        open(os.path.join(pack_dir, "redirects.pickle"), "rb"))
    loaded_resource = Resources()
    loaded_resource.update(redirects=redirect_map)

    complete_and_tokens()
