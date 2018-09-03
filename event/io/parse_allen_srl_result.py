import os
import sys
import json


def get_srl(verb_data):
    spans = {}

    tags = verb_data['tags']

    t = None

    for index, tag in enumerate(tags):
        if tag.startswith('B'):
            if t:
                # Output last span.
                spans[t] = (start, end)
            t = tag.split('-', 1)[1]
            start = index
            end = index
        elif tag.startswith('I'):
            end = index
        elif tag == 'O':
            if t:
                spans[t] = (start, end)

        if index == len(tags) - 1:
            if t:
                spans[t] = (start, end)

    return spans


def align_to_char_span(sent, span, span_words):
    print(sent)
    print(span)
    print(span_words)


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    out_dir = sys.argv[3]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(input_file) as inf, open(output_file) as outf:
        data = []

        for inline, outline in zip(inf, outf):
            input_data = json.loads(inline)
            output_data = json.loads(outline)
            docid = input_data['docid']
            start = input_data['start']
            end = input_data['end']
            sent = input_data['sentence']

            output_words = output_data['words']

            for verb_data in output_data['verbs']:
                spans = get_srl(verb_data)

                for span in spans:
                    span_words = output_words[span[0], span[1] + 1]
                    align_to_char_span(sent, span, span_words)
                    input('check')
