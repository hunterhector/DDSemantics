import os
import sys
import json
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


def get_srl(verb_data):
    args = {}

    tags = verb_data['tags']

    t = None

    for index, tag in enumerate(tags):
        if tag.startswith('B'):
            if t:
                # Output last span.
                args[t] = (start, end)
            t = tag.split('-', 1)[1]
            start = index
            end = index
        elif tag.startswith('I'):
            end = index
        elif tag == 'O':
            if t:
                args[t] = (start, end)

        if index == len(tags) - 1:
            if t:
                args[t] = (start, end)

    return args


def align_to_char_span(sent, span, span_words):
    print(sent)
    print(span)
    print(span_words)


def write_out(out_dir, docid, data):
    with open(os.path.join(out_dir, docid + '.json', 'w')) as out:
        json.dump(data, out)


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    out_dir = sys.argv[3]

    # How to reproduce their tokens? Run their tokenizer!
    tokenizer = SpacyWordSplitter(language='en_core_web_sm',
                                  pos_tags=True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(input_file) as inf, open(output_file) as outf:
        data = []

        lastid = None

        for inline, outline in zip(inf, outf):
            input_data = json.loads(inline)
            output_data = json.loads(outline)
            docid = input_data['docid']
            start = input_data['start']
            end = input_data['end']
            sent = input_data['sentence']

            output_words = output_data['words']

            tokens = tokenizer.split_words(sent)
            words = [(token.text, token.idx) for token in tokens]

            print(output_words)
            print(words)

            for verb_data in output_data['verbs']:
                args = get_srl(verb_data)

                for arg_type, span in args.items():
                    span_words = words[span[0]: span[1] + 1]

                    align_to_char_span(tokenizer, sent, span, span_words)
                    input('check')

            if lastid and not docid == lastid:
                write_out(out_dir, lastid, [])

            lastid = docid

        if lastid:
            write_out(out_dir, lastid, [])
