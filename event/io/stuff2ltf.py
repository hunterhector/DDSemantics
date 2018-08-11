import sys
import os
import json
import xml.etree.ElementTree as ET


class LTF:
    def __init__(self, language='eng'):
        root = ET.Element('xml')
        lctl_node = ET.SubElement(root, 'LCTL_TEXT')
        lctl_node.set('lang', language)
        doc_node = ET.SubElement(lctl_node, 'DOC')
        self.text_node = ET.SubElement(doc_node, 'TEXT')
        self.text_offset = 0

    def add_seg(self):
        seg = ET.SubElement(self.text_node, 'SEG')
        seg.set('start_char', self.text_offset)
        return seg

    def add_token(self, seg):
        pass


def sausage_to_ltf(in_file_path, out_file_path):
    with open(in_file_path) as inf, open(out_file_path) as out:
        print(in_file_path)
        sausage_json = json.load(inf)

        for obj in sausage_json:
            result = obj['result']

            best_words = []
            for hypos in result['sausage']:
                best_words.append((hypos[0]['word'], hypos[0]['confidence']))


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    format = sys.argv[3]

    for f in os.listdir(input_dir):
        file_path = os.path.join(input_dir, f)

        if not file_path.endswith('.json'):
            continue

        if format == 'sausage':
            sausage_to_ltf(file_path)
        else:
            raise ValueError("Unsupport format: " + format)
