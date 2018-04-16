import sys
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.stem import WordNetLemmatizer

in_dir = sys.argv[1]
out_file = sys.argv[2]
lemmatizer = WordNetLemmatizer()

with open(out_file, 'w') as out:
    for file in os.listdir(in_dir):
        tree = ET.parse(os.path.join(in_dir, file))
        root = tree.getroot()
        doc = root[0]

        docid = file.split(".")[0]

        out.write("# newdoc id = %s\n" % docid)

        for text in doc:
            sent_id = 0

            for seg in text:
                sent_id += 1
                token_id = 0

                words = []
                spans = []
                for tokens in seg:
                    if tokens.tag == 'TOKEN':
                        words.append(tokens.text)
                        begin = int(tokens.attrib['start_char']) - 1
                        end = int(tokens.attrib['end_char'])
                        spans.append(
                            "%d,%d" % (begin, end)
                        )

                word_pos = nltk.pos_tag(words)

                l_tokens = []
                for (word, pos), span in zip(word_pos, spans):
                    token_id += 1
                    l_tokens.append([token_id, word, lemmatizer.lemmatize(word),
                                   "_", pos, "_", "_", "_", "_", "_", span])

                out.write("# sent_id = %d\n" % sent_id)

                for tokens in l_tokens:
                    out.write("\t".join([str(t) for t in tokens]))
                    out.write('\n')
                out.write("\n")
