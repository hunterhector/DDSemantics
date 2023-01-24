import pdb
from collections import defaultdict, Counter

from forte.common import Resources, Config
from forte.data import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence

from onto.facets import EventMention, EventArgument, EntityMention


class ImplicitStats(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.__argument_distances = Counter()
        self.__simple_counts = Counter()

    def _process(self, input_pack: DataPack):
        self.__simple_counts["number of docs"] += 1

        all_sents = []
        entity_position = {}

        sentence: Sentence
        for i, sentence in enumerate(input_pack.get(Sentence)):
            self.__simple_counts["number of sentences"] += 1

            all_sents.append(sentence)
            for entity_mention in sentence.get(EntityMention):
                entity_position[entity_mention.tid] = i

        for sent_num, sentence in enumerate(all_sents):
            event_mention: EventMention
            for event_mention in sentence.get(EventMention):
                self.__simple_counts["number of events"] += 1
                arguments = input_pack.get_links_by_parent(event_mention)

                arg: EventArgument
                for arg in arguments:
                    self.__simple_counts["number of arguments"] += 1
                    arg_position = entity_position[arg.get_child().tid]
                    self.__argument_distances[arg_position - sent_num] += 1

    def finish(self, resource: Resources):
        print("ImplicitStats processor finished.")
        print("===Summary of statistics===")
        for k, v in self.__simple_counts.items():
            print(f"{k}:{v}")

        print("Distance distribution of implicit arguments:")
        for k, v in self.__argument_distances.items():
            print(f"{k}:{v}")
