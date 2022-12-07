import os
import re
import xml.etree.ElementTree as ET
from typing import Iterator

from facets.html_parser import CustomHTMLParser
from forte.data import DataPack, Span
from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader
from forte.data.types import ReplaceOperationsType
from onto.facets import EventMention, EventArgument, Hopper, EntityMention

parser = CustomHTMLParser()


def parse_ere_source_tags(text):
    parser.reset()
    parser.feed(text)

    line_offsets = [0] + [m.start() + 1 for m in re.finditer("\n", text)]

    tag_spans = []

    begin_tags, end_tags = parser.get_tags()

    for (b_tag_name, (l_b_tag, col_b_tag), begin_tag_text) in begin_tags:
        b_tag_begin = line_offsets[l_b_tag - 1] + col_b_tag
        b_tag_end = b_tag_begin + len(begin_tag_text)

        tag_spans.append((
            (b_tag_begin, b_tag_end),
            " " * (b_tag_end - b_tag_begin)
        ))

    for (
            e_tag_name,
            (l_e_tag, col_e_tag)
    ) in end_tags:
        e_tag_begin = line_offsets[l_e_tag - 1] + col_e_tag
        e_tag_end = e_tag_begin + len(e_tag_name) + 3

        end_tag_span = (e_tag_begin, e_tag_end)
        replacement = " " * (e_tag_end - e_tag_begin)

        if e_tag_name.lower() == "headline":
            if text[e_tag_begin - 1].isspace():
                replacement = "." + replacement
                end_tag_span = (e_tag_begin - 1, e_tag_end)
            else:
                raise RuntimeError("Cannot handle a specific headline well.")

        tag_spans.append(
            (
                end_tag_span,
                replacement
            )
        )


    return tag_spans


unicode_replace = {
    u"\u0080": " ",
    u"\u0096": " ",
    u"\u0097": " ",
    u"\u0095": " ",
    u"\u0091": " ",
    u"\u0092": "'",
    u"\u0093": '"',
    u"\u0094": '"',
    u"\u00A7": " ",  # section marker.
    u"\u00a0": " ",
    u"\u2019": "'",
    u"\u200f": " ",
}


def replace_unicode(text: str):
    res = text
    for k, v in unicode_replace.items():
        res = res.replace(k, v)

    assert len(res) == len(text)
    return res


class EREReader(PackReader):
    def _collect(self, input_dirs) -> Iterator[str]:
        for root_dir, src_dir, ere_dir in input_dirs:
            for file_name in os.listdir(os.path.join(root_dir, ere_dir)):
                for ere_ext in self.configs.ere_ext:
                    if file_name.endswith(ere_ext):
                        base_name = file_name.replace(ere_ext, "")
                        for src_ext in self.configs.src_ext:
                            src_file = os.path.join(
                                root_dir, src_dir, base_name + src_ext
                            )
                            if os.path.exists(src_file):
                                yield (
                                    src_file,
                                    os.path.join(root_dir, ere_dir, file_name)
                                )
                                break
                        else:
                            raise ValueError(
                                f"Cannot find file [{base_name}] in directory"
                                f"[{src_dir}] for all extensions.")

    @classmethod
    def default_configs(cls):
        return {
            "src_ext": [".xml", ".txt", ".cmp.txt"],
            "ere_ext": [".rich_ere.xml", ".event_hoppers.xml"]
        }

    def text_replace_operation(self, text: str) -> ReplaceOperationsType:
        more_patterns = ["</DOC"]
        matched = {}

        # Find the html tags.
        replace_spans = []
        for (b, e), replacement in parse_ere_source_tags(text):
            replace_spans.append((Span(b, e), replacement))

            for p in more_patterns:
                if p in text[b: e]:
                    try:
                        matched[p].append((b, e))
                    except KeyError:
                        matched[p] = [(b, e)]

        # Find additional patterns.
        additional_spans = []
        for pattern in more_patterns:
            p_len = len(pattern)
            for b in [m.start() for m in re.finditer(pattern, text)]:
                additional_spans.append(Span(b, b + p_len))

        for additional_span in additional_spans:

            for r_span, _ in replace_spans:
                if r_span.begin <= additional_span.begin < r_span.end:
                    break
            else:
                p_len = additional_span.end - additional_span.begin
                # This means this pattern is not already matched before.
                replace_spans.append((additional_span, " " * p_len))

        return replace_spans

    def _parse_pack(self, input_paths) -> Iterator[PackType]:
        src_path, ere_path = input_paths

        with open(ere_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

            pack = DataPack()

            with open(src_path) as src_text_file:
                src_text = replace_unicode(src_text_file.read())
                self.set_text(pack, src_text)
                assert len(src_text) == len(pack.text)

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
                            evm.event_type = em_node.get(
                                "type") + "_" + em_node.get("subtype")
                            evm.realis = em_node.get("realis")
                            evm.audience = em_node.get("audience")
                            evm.formality = em_node.get("formality")
                            evm.medium = em_node.get("medium")
                            evm.schedule = em_node.get("schedule")
                            evm.id = em_node.get("id")

                            evm_arg = em_node.find("em_arg")

                            if evm_arg is not None:
                                arg_mention_id = evm_arg.get(
                                    "entity_mention_id")
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
