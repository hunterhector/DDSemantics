import sys
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer


def conllu2txt(in_file, out_dir):
    with open(in_file) as conllu:
        for line in conllu:
            if line.startswith("#"):
                if line.startswith('# new doc'):
                    docid = line.split('=')[1].strip()
                    text = ""
                elif line.startswith('# sent_id'):
                    pass
            else:
                parts = line.split('\t')
                lemma = parts[2]

def ltf2txt(in_dir, out_dir):
    lemmatizer = WordNetLemmatizer()

    for file in os.listdir(in_dir):
        tree = ET.parse(os.path.join(in_dir, file))
        root = tree.getroot()
        doc = root[0]

        docid = file.split(".")[0]

        out_file = os.path.join(out_dir, docid + '.txt')
        sent_file = os.path.join(out_dir, docid + '.sent')

        with open(out_file, 'w') as out, open(sent_file, 'w') as sent_out:
            text = ""

            for entry in doc:
                bad_ending = False
                sents = []

                for seg in entry:
                    new_sent = True

                    sents.append(
                        "%d %s" % (
                            int(seg.attrib['start_char']) - 1,
                            seg.attrib['end_char']
                        )
                    )

                    for token in seg:
                        if token.tag == 'TOKEN':
                            begin = int(token.attrib['start_char']) - 1

                            if new_sent:
                                if bad_ending and begin > len(text):
                                    text += '.'
                                if begin > len(text):
                                    text += ('\n' * (begin - len(text)))
                            else:
                                text += (' ' * (begin - len(text)))

                            text += token.text

                            bad_ending = not token.attrib['pos'] == 'punct'
                            new_sent = False
            out.write(text)
            sent_out.write('\n'.join(sents))


def main():
    in_dir = sys.argv[1]
    out_file = sys.argv[2]
    format_in = sys.argv[3]

    if format_in == 'ltf':
        ltf2txt(in_dir, out_file)


if __name__ == '__main__':
    main()
