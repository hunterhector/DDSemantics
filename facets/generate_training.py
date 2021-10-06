import os
import pickle
import re
import sys
from typing import Dict, Any, Set

from facets.wiki.extractor.sentence_sim import sentence_clues
from ft.onto.wikipedia import WikiAnchor
from smart_open import open

from forte import Pipeline
from forte.common import Resources
from forte.common.configuration import Config
from forte.data import DataPack
from forte.data.readers import DirPackReader
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Subword, Sentence


def filter_out(title: str):
    return re.match(r"(List_of_.+)|"
                    r"(Index_of_.+)|"
                    r"(Outline_of_.+)|"
                    r"(.*\(disambiguation\).*)", title)


def find_best_sent(input_sent: Sentence, target_pack: DataPack,
                   src_pack: DataPack):
    count = 0
    data = ["input: " + input_sent.text + " src page: " + src_pack.pack_name]

    clues = sentence_clues(input_sent, src_pack.pack_name, target_pack)

    for c in clues:
        if c[0]:
            import pdb
            pdb.set_trace()

    # sent: Sentence
    # for sent in target_pack.get(Sentence):
    #     anchors = [a.target_page_name for a in target_pack.get(WikiAnchor,
    #     sent)]
    #
    #     count += 1
    #     data.append("target: " + sent.text + " [anchors] " + " ".join(
    #     anchors))
    #
    #     if count % 5 == 0:
    #         print("\n".join(data))
    #         import pdb
    #         pdb.set_trace()


class TrainingGenerator(PackProcessor):
    """
    Create training data from subword tokenized data.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.indices = {}
        with open(self.configs.article_index_path) as article_index:
            for line in article_index:
                page_name, path = line.strip().split()
                self.indices[page_name] = path

    def _process(self, input_pack: DataPack):
        if filter_out(input_pack.pack_name):
            return

        sentence: Sentence
        subword: Subword
        anchor: WikiAnchor
        for sentence in input_pack.get(Sentence):
            print('sent is', sentence.text)
            for anchor in input_pack.get(WikiAnchor, sentence):
                print('find anchor for ', anchor.text)
                self.find_target_sent(sentence, anchor, input_pack)

    def find_target_sent(self, src_sent: Sentence, target: WikiAnchor,
                         src_pack: DataPack):
        anchor_target = target.target_page_name
        anchor_target = self.resources.get(
            "redirects").get(anchor_target, anchor_target)

        if anchor_target not in self.indices:
            # Return some empty stuff
            return

        target_path = os.path.join(
            self.configs.data_pack_dir, self.indices[target.target_page_name])
        target_pack = DataPack.deserialize(
            target_path, serialize_method="pickle", zip_pack=True)
        find_best_sent(src_sent, target_pack, src_pack)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs = super().default_configs()
        configs.update({
            "data_pack_dir": None,
            "article_index_path": None
        })
        return configs


if __name__ == '__main__':
    pack_dir = sys.argv[1]
    pack_input = os.path.join(pack_dir, "nif_raw_struct_links_token")
    index_path = os.path.join(pack_input, "article.idx")

    redirect_map = pickle.load(
        open(os.path.join(pack_dir, "redirects.pickle"), "rb"))
    loaded_resource = Resources()
    loaded_resource.update(redirects=redirect_map)
    print("Finish loading redirects.")

    Pipeline(loaded_resource).set_reader(
        DirPackReader(),
        config={
            "suffix": ".pickle.gz",
            "zip_pack": True,
            "serialize_method": "pickle",
        }
    ).add(
        TrainingGenerator(),
        config={
            "data_pack_dir": pack_input,
            "article_index_path": index_path,
        }
    ).run(pack_input)
