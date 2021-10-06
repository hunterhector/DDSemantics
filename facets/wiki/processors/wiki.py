import logging
import sys
from collections import defaultdict
from typing import List

from forte.data import DataPack
from forte.datasets.wikipedia.dbpedia import WikiPackReader
from forte.datasets.wikipedia.dbpedia.db_utils import state_type, \
    get_resource_name
from forte.processors.base import PackProcessor
from ft.onto.wikipedia import WikiAnchor, WikiPage, WikiArticleTitle

from flashtext import KeywordProcessor


class TempProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        page = input_pack.get_single(WikiPage)
        sys.stdout.write(".")
        sys.stdout.flush()
        if input_pack.get_single(WikiPage).page_id == '729678636':
            import pdb
            pdb.set_trace()


class WikiDuplicateLinksDedup(PackProcessor):
    """
    Created some links by accident, deduplicate with this.
    """

    def _process(self, input_pack: DataPack):
        all_anchors = defaultdict(list)
        anchor: WikiAnchor
        for anchor in input_pack.get(WikiAnchor):
            all_anchors[(anchor.span.begin, anchor.span.end)].append(anchor)

        for span in all_anchors.keys():
            l_a: List[WikiAnchor] = all_anchors[span]
            if len(l_a) > 1:
                if len(l_a) > 2:
                    print(input_pack.pack_name, l_a[0].target_page_name,
                          len(l_a))
                    logging.error(
                        "There are links that have more than 2 copies.")
                    import pdb
                    pdb.set_trace()
                for a in l_a[1:]:
                    # Removing duplicates.
                    input_pack.delete_entry(a)


class WikiAddTitle(PackProcessor):
    """
    Add Wikipedia title to the end.
    """

    def _process(self, input_pack: DataPack):
        title_text = input_pack.get_single(WikiPage).page_name
        new_text = input_pack.text + "\n" + title_text
        title_begin = len(input_pack.text) + 1
        title_end = title_begin + len(title_text)
        input_pack.set_text(new_text)
        WikiArticleTitle(input_pack, title_begin, title_end)


class WikiEntityCompletion(PackProcessor):
    """
    Create more anchors by repeating the entry annotation in the page.
    """

    def _process(self, input_pack: DataPack):
        kp = KeywordProcessor(case_sensitive=True)
        anchor_entities = {}
        existing_anchors = set()

        anchor: WikiAnchor
        for anchor in input_pack.get(WikiAnchor):
            kp.add_keyword(anchor.text)
            existing_anchors.add((anchor.span.begin, anchor.span.end))

            try:
                anchor_entities[anchor.text].append(anchor)
            except KeyError:
                anchor_entities[anchor.text] = [anchor]

        for kw, b, e in kp.extract_keywords(input_pack.text, span_info=True):
            targets = anchor_entities[kw]

            if (b, e) in existing_anchors:
                # Ignore existing anchors.
                continue

            copy_from: WikiAnchor
            if len(targets) == 1:
                copy_from = targets[0]
            elif len(targets) > 1:
                latest_ = targets[0]
                for t in targets:
                    if t.begin < b:
                        latest_ = t
                copy_from = latest_
            else:
                raise RuntimeError(f"Unknown target length {len(targets)}")

            anchor = WikiAnchor(input_pack, b, e)
            anchor.target_page_name = copy_from.target_page_name
            anchor.is_external = copy_from.is_external
            input_pack.add_entry(anchor)
