import os
import sys
import json
from itertools import izip


def get_srl(verb_data):
    spans = {}

    tags = verb_data['tags']

    t = None

    for index, tag in enumerate(tags):
        if tag.startswith('B'):
            t = tag.split('-', 1)[1]
            start = index
            end = index
        elif tag.startswith('I'):
            end = index
        elif tag == 'O':
            if t:
                spans[t] = (start, end)

    return spans


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    out_dir = sys.argv[3]

    with open(input_file) as inf, open(output_file) as outf:

        data = []

        for inline, outline in izip(inf, outf):
            input_data = json.loads(inline)
            output_data = json.loads(outline)
            docid = input_data['docid']
            start = input_data['start']
            end = input_data['end']
            sent = input_data['sentence']

            for verb_data in outline['verbs']:
                spans = get_srl(verb_data)
                print(sent)
                print(spans)
                input('check')
