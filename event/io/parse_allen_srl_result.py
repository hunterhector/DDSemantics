import os
import sys
import json
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


def get_srl(verb_data):
    args = {}

    tags = verb_data["tags"]

    t = None

    # C_ARG0 means discontinous argument
    # "R-" arguments are arguments that are referencing another argument in the sentence.
    # "R" may be simply ignored?

    for index, tag in enumerate(tags):
        if tag.startswith("B"):
            if t:
                # Output last span.
                args[t] = (start, end)
            t = tag.split("-", 1)[1]
            start = index
            end = index
        elif tag.startswith("I"):
            end = index
        elif tag == "O":
            if t:
                args[t] = (start, end)

        if index == len(tags) - 1:
            if t:
                args[t] = (start, end)

    return args


def write_out(out_dir, docid, data):
    with open(os.path.join(out_dir, docid + ".json"), "w") as out:
        json.dump(data, out)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    out_dir = sys.argv[3]

    # How to reproduce their tokens? Run their tokenizer!
    tokenizer = SpacyWordSplitter(language="en_core_web_sm", pos_tags=True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(input_file) as inf, open(output_file) as outf:
        data = {"srl": []}

        lastid = None

        for inline, outline in zip(inf, outf):
            input_data = json.loads(inline)
            output_data = json.loads(outline)
            docid = input_data["docid"]

            if lastid and not docid == lastid:
                data["docid"] = lastid

                write_out(out_dir, lastid, data)
                data = {"srl": []}

            sent_start = input_data["start"]
            sent = input_data["sentence"]

            tokens = tokenizer.split_words(sent)
            word_spans = [token.idx for token in tokens]

            for verb_data in output_data["verbs"]:
                srl = []

                args = get_srl(verb_data)
                for arg_type, span in args.items():
                    s = word_spans[span[0]]
                    e = word_spans[span[1]] + len(tokens[span[1]])
                    srl.append(
                        {
                            "role": arg_type,
                            "text": sent[s:e],
                            "span": (sent_start + s, sent_start + e),
                        }
                    )

                if len(srl) > 1:
                    # Keep the ones with arguments.
                    data["srl"].append(srl)
            lastid = docid

        if lastid:
            write_out(out_dir, lastid, data)
