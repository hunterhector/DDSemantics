"""This extractor takes the string from wikipedia info-box text"""
from collections import Iterable

from forte.data import BaseExtractor, DataPack
from forte.data.ontology.core import EntryType
from ft.onto.wikipedia import WikiInfoBoxProperty, WikiInfoBoxMapped


class InfoboxExtractor(BaseExtractor):
    def __int__(self, config):
        super.__init__(config)

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update({"infobox_type": "mapped"})
        return config

    def extract(self, pack: DataPack, instance=None):
        for info_box in self.__get_info_boxes(pack):
            self.add(info_box.key)

    def __get_info_boxes(self, pack: DataPack) -> Iterable[EntryType]:
        if self.config.infobox_type == "property":
            yield from pack.get(WikiInfoBoxProperty)
        elif self.config.infobox_type == "mapped":
            yield from pack.get(WikiInfoBoxMapped)
        else:
            yield from pack.get(WikiInfoBoxProperty)
            yield from pack.get(WikiInfoBoxMapped)
