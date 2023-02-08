from typing import Dict, Any, List

import ipdb
from collections import defaultdict, Counter

from forte.common import Resources, Config
from forte.data import DataPack, MultiPack, Selector
from forte.processors.base import PackProcessor, MultiPackProcessor
from ft.onto.base_ontology import Sentence

from transformers import T5Tokenizer, T5ForConditionalGeneration

from facets.common.utils import color_print
from onto.facets import EventMention, EventArgument, EntityMention, CopyLink


class DetectionProcessor(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base", device_map="auto"
        )

    def _process(self, input_pack: DataPack):
        print(input_pack.text)
        input_ids = self.tokenizer(input_pack.text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids)
        self.tokenizer.decode(outputs[0])

    def _qa_style_processing(self):
        pass


class ArgumentRoleDetection(MultiPackProcessor):
    """
        Given a gold event, find the arguments that need to be filled.
    """

    def _process(self, input_pack: MultiPack):
        for name, pack in input_pack.iter_packs():
            if not name == "default":
                self.__use_gold_roles(
                    input_pack,
                    input_pack.get_pack("default"),
                    pack
                )

    def __use_gold_roles(
            self,
            input_pack: MultiPack,
            gold_pack: DataPack,
            context_pack: DataPack
    ):
        def get_copied_from(mention: EventMention):
            for link in input_pack.get_links_by_child(mention):
                if isinstance(link, CopyLink):
                    return link.get_parent()

        m = context_pack.get_single(EventMention)
        source_m = get_copied_from(m)

        arg_link: EventArgument
        for arg_link in gold_pack.get_links_by_parent(source_m):
            if isinstance(arg_link, EventArgument):
                arg_to_fill = EventArgument(context_pack, m, None)
                arg_to_fill.vb_role = arg_link.vb_role
                arg_to_fill.pb_role = arg_link.pb_role
                arg_to_fill.role = arg_link.role
                context_pack.add_entry(arg_to_fill)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """

        Returns:
            `pack_prefix` is the prefix of data packs

        """
        return {
            "pack_prefix": "context_"
        }


class ContextProcessor(MultiPackProcessor):
    """
        This processor creates a new pack for each event, including
        only the context for that event.
    """

    def _process(self, pack: MultiPack):
        src_pack: DataPack = pack.get_pack("default")

        all_sents: List[Sentence] = []
        entity_sent_pos = {}

        sentence: Sentence
        for i, sentence in enumerate(src_pack.get(Sentence)):
            all_sents.append(sentence)
            for entity_mention in sentence.get(EntityMention):
                entity_sent_pos[entity_mention.tid] = i

        sentence: Sentence
        context_id = 0
        for sent_num, sentence in enumerate(src_pack.get(Sentence)):
            event_mention: EventMention
            for event_mention in sentence.get(EventMention):
                self._build_context(
                    pack,
                    context_id,
                    all_sents,
                    event_mention,
                    sent_num,
                )

    def _build_context(
            self,
            pack: MultiPack,
            context_id: int,
            sentences: List[Sentence],
            event_mention: EventMention,
            event_sentence_num: int,
    ):
        context_pack: DataPack = pack.add_pack(ref_name=f"context_{context_id}")

        first_sent = max(0, event_sentence_num - 2)
        last_sent = min(len(sentences) - 1, event_sentence_num + 2)
        event_sent_new_pos = event_sentence_num - first_sent

        text_ = ""
        splitter = ""

        # beginning and endings of the sentences
        bes = []

        for i in range(first_sent, last_sent):
            text_ += splitter
            text_ += sentences[i].text

            bes.append((len(text_) - len(sentences[i].text), len(text_)))
            splitter = "\n"

        context_pack.set_text(text_)

        in_sent_begin = event_mention.begin - sentences[event_sentence_num].begin
        in_sent_end = event_mention.end - sentences[event_sentence_num].begin

        new_begin = in_sent_begin + bes[event_sent_new_pos][0]
        new_end = in_sent_end + bes[event_sent_new_pos][0]

        new_mention = EventMention(context_pack, new_begin, new_end)
        context_pack.add_entry(new_mention)

        pack.add_entry(CopyLink(pack, event_mention, new_mention))


class MatchingProcessor(MultiPackProcessor):
    """
        Match the generated entities to get the implicit roles.
        - Direct matching or Coreference
    """

    def _process(self, pack: MultiPack):
        for pack_name, p in pack.iter_packs():
            if pack_name is not "default":
                # do coref
                pass

    def coref(self, pack: DataPack):
        pass


class RamsOutputWriter(MultiPackProcessor):
    """
        Write out the RAMS format data.
    """

    def _process(self, input_pack: MultiPack):
        pass

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "output_dir": None
        }


class ImplicitStats(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.__argument_distances = Counter()
        self.__simple_counts = Counter()

    def _process(self, input_pack: DataPack):
        self.__simple_counts["number of docs"] += 1

        all_sents = []
        entity_sent_pos = {}

        sentence: Sentence
        for i, sentence in enumerate(input_pack.get(Sentence)):
            self.__simple_counts["number of sentences"] += 1

            all_sents.append(sentence)
            for entity_mention in sentence.get(EntityMention):
                entity_sent_pos[entity_mention.tid] = i

        for sent_num, sentence in enumerate(all_sents):
            event_mention: EventMention
            for event_mention in sentence.get(EventMention):
                self.__simple_counts["number of events"] += 1
                arguments = input_pack.get_links_by_parent(event_mention)

                print(f"Event mention: {event_mention.text} in [Sent {sent_num}]: ")

                color_print(
                    sentence.text,
                    [
                        (
                            event_mention.begin - sentence.begin,
                            event_mention.end - sentence.begin
                        )
                    ],
                    ["red"]
                )

                print(f"It contains {len(arguments)} arguments.")

                arg: EventArgument
                for arg in arguments:
                    self.__simple_counts["number of arguments"] += 1
                    arg_em: EntityMention = arg.get_child()

                    arg_sent_num = entity_sent_pos[arg_em.tid]
                    self.__argument_distances[arg_sent_num - sent_num] += 1

                    arg_sent = all_sents[arg_sent_num]

                    if not arg_sent_num == sent_num:
                        print(f"Argument role {arg.vb_role} in [Sent {arg_sent_num}]: ", end="")
                        color_print(
                            arg_sent.text,
                            [
                                (
                                    arg_em.begin - arg_sent.begin,
                                    arg_em.end - arg_sent.begin
                                )
                            ],
                            ["green"]
                        )
                import pdb
                pdb.set_trace()

    def finish(self, resource: Resources):
        print("ImplicitStats processor finished.")
        print("===Summary of statistics===")
        for k, v in self.__simple_counts.items():
            print(f"{k}:{v}")

        print("Distance distribution of implicit arguments:")
        for k, v in self.__argument_distances.items():
            print(f"{k}:{v}")
