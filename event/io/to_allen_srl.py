import sys
import os
import json


def convert(docid, inf, sent, out):
    txt = inf.read()

    for line in sent:
        start, end, sid, *_ = line.split()
        start = int(start)
        end = int(end)
        sent_json = {
            "sentence": txt[start:end],
            "docid": docid,
            "start": start,
            "end": end,
        }
        out.write(json.dumps(sent_json))
        out.write("\n")


if __name__ == "__main__":
    txt_dir = sys.argv[1]
    out_file = sys.argv[2]

    with open(out_file, "w") as out:
        for fname in os.listdir(txt_dir):
            if fname.endswith("txt"):
                docid = fname.replace(".txt", "")
                sent_file = fname.replace(".txt", ".sent")
                with open(os.path.join(txt_dir, fname)) as inf, open(
                    os.path.join(txt_dir, sent_file)
                ) as sent:
                    convert(docid, inf, sent, out)
