import os
from typing import Iterator
import xml.etree.ElementTree as ET

from forte.data import DataPack
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from onto.facets import EventMention, EventArgument, Hopper, EntityMention


class EREReader(PackReader):
    def _collect(self, input_dirs) -> Iterator[str]:
        for d in input_dirs:
            for file_name in os.listdir(os.path.join(d, self.configs.ere_dir)):
                if file_name.endswith(self.configs.ere_ext):
                    base_name = file_name.replace(self.configs.ere_ext, "")
                    src_file = os.path.join(
                        d, self.configs.src_dir,
                        base_name + self.configs.src_ext
                    )

                    yield (
                        src_file,
                        os.path.join(d, self.configs.ere_dir, file_name)
                    )

    @classmethod
    def default_configs(cls):
        return {
            "src_dir": "source",
            "ere_dir": "ere",
            "src_ext": ".xml",
            "ere_ext": ".rich_ere.xml"
        }

    def _parse_pack(self, input_paths) -> Iterator[PackType]:
        src_path, ere_path = input_paths

        with open(ere_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

            pack = DataPack()

            with open(src_path) as src_text:
                pack.set_text(src_text.read())

            pack.pack_name = root.get("doc_id")

            args = []
            ems = {}

            for c1 in root:
                if c1.tag == "hoppers":
                    for c2 in c1.iter("hopper"):
                        hopper = Hopper(pack)
                        hopper.id = c2.get("id")

                        for em_node in c2.iter("event_mention"):
                            trigger = em_node.find("trigger")
                            begin = int(trigger.get("offset"))
                            end = begin + int(trigger.get("length"))
                            evm = EventMention(pack, begin, end)
                            evm.event_type = em_node.get("type") + "_" + em_node.get("subtype")
                            evm.realis = em_node.get("realis")
                            evm.audience = em_node.get("audience")
                            evm.formality = em_node.get("formality")
                            evm.medium = em_node.get("medium")
                            evm.schedule = em_node.get("schedule")
                            evm.id = em_node.get("id")

                            evm_arg = em_node.find("em_arg")

                            arg_mention_id = evm_arg.get("entity_mention_id")
                            if arg_mention_id is None:
                                arg_mention_id = evm_arg.get("filler_id")

                            args.append(
                                (
                                    evm,
                                    (
                                        arg_mention_id,
                                        evm_arg.get("role"),
                                        evm_arg.get("realis"),
                                        evm_arg.get("id"),
                                    )
                                )
                            )

                            hopper.add_member(evm)
                elif c1.tag == "entities":
                    for c2 in c1.iter("entity"):
                        for em_node in c2.iter("entity_mention"):
                            begin = int(em_node.get("offset"))
                            end = begin + int(em_node.get("length"))
                            em = EntityMention(pack, begin, end)
                            em.ner_type = c1.get("type")
                            em.id = c1.get("id")
                            em.is_filler = False

                            ems[em_node.get("id")] = em
                elif c1.tag == "fillers":
                    for filler in c1.iter("filler"):
                        begin = int(filler.get("offset"))
                        end = begin + int(filler.get("length"))
                        em = EntityMention(pack, begin, end)
                        em.ner_type = c1.get("type")
                        em.id = c1.get("id")
                        em.is_filler = True

                        ems[filler.get("id")] = em

            for evm, (em_id, role, realis, arg_id) in args:
                em = ems[em_id]
                argument = EventArgument(pack, evm, em)
                argument.role = role
                argument.realis = realis
                argument.id = arg_id

            yield pack

