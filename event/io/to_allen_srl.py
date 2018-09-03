import sys
import os
import json


def convert(inf, sent, out):
    for line in inf:
        sent_json = {'sentence': line.strip()}
        out.write(json.dumps(sent_json))

    out.write('\n')


if __name__ == '__main__':
    txt_dir = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname in os.listdir(txt_dir):
        if fname.endswith('txt'):
            sent_file = fname.replace('.txt', '.sent')
            with open(os.path.join(txt_dir, fname)) as inf, open(
                    os.path.join(txt_dir, sent_file)) as sent, open(
                os.path.join(out_dir, fname), 'w') as out:
                convert(inf, sent, out)
