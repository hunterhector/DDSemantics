import glob
import os
import xml.etree.ElementTree as ET

from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
)


class FrameNet(DataLoader):
    def __init__(self, params, corpus, with_doc=False):
        super().__init__(params, corpus, with_doc)
        self.params = params
        self.ns = {
            "icsi": "http://framenet.icsi.berkeley.edu",
        }

    def parse_full_text(self, full_text_file, doc):
        root = ET.parse(full_text_file).getroot()

        full_text = ""
        offset = 0

        annotations = []

        for sent in root.findall("icsi:sentence", self.ns):
            sent_text = sent.find("icsi:text", self.ns).text

            full_text += sent_text
            full_text += "\n"

            for anno_set in sent.findall("icsi:annotationSet", self.ns):
                targets = []
                fes = []

                if not "frameName" in anno_set.attrib:
                    continue

                frame_name = anno_set.attrib["frameName"]

                for layer in anno_set.findall("icsi:layer", self.ns):
                    layer_type = layer.attrib["name"]

                    if layer_type == "Target":
                        label = layer.find("icsi:label", self.ns)

                        if label is not None:
                            s = int(label.attrib["start"])
                            e = int(label.attrib["end"]) + 1
                            text = sent_text[s:e]
                            targets.append((s + offset, e + offset, text))
                    elif layer_type == "FE":
                        for label in layer.findall("icsi:label", self.ns):
                            label_name = label.attrib["name"]

                            if "itype" in label.attrib:
                                # Null instantiation.
                                pass
                            else:
                                s = int(label.attrib["start"])
                                e = int(label.attrib["end"]) + 1
                                text = sent_text[s:e]
                                fes.append((s + offset, e + offset, text, label_name))

                if targets:
                    max_len = 0
                    target = None
                    for i, (s, e, text) in enumerate(targets):
                        if e - s > max_len:
                            max_len = e - s
                            target = s, e, text

                    annotations.append((frame_name, target, fes))

            offset = len(full_text)

        doc.set_text(full_text)

        for frame_name, target, fes in annotations:
            ev = doc.add_hopper()
            target_start, target_end, text = target
            evm = doc.add_predicate(
                ev, Span(target_start, target_end), text=text, frame_type=frame_name
            )

            for start, end, fe_text, role in fes:
                filler = doc.add_filler(Span(start, end), fe_text)
                doc.add_argument_mention(evm, filler.aid, role)

        return doc

    def get_doc(self):
        super().get_doc()

        full_text_dir = os.path.join(self.params.fn_path, "fulltext")

        for full_text_path in glob.glob(full_text_dir + "/*.xml"):
            doc = DEDocument(self.corpus)
            docid = os.path.basename(full_text_path).replace(".xml", "")
            doc.set_id(docid)
            self.parse_full_text(full_text_path, doc)
            yield doc
