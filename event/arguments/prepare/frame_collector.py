import logging
import os
import operator
from collections import Counter, defaultdict
import json
import gzip


def open_func(data_in):
    if data_in.endswith('.gz'):
        return gzip.open
    else:
        return open


class ArgFrameCollector:
    def __init__(self):
        self.args_fe_map = defaultdict(Counter)
        self.fe_args_map = defaultdict(Counter)
        self.fe_count = Counter()
        self.arg_count = Counter()
        self.len_arg_fields = 5

    def read_dataset(self, data_in):
        batch = 10000
        count = 0

        logging.info("Reading: " + data_in)

        num_format_errors = 0

        with open_func(data_in)(data_in, 'rt') as data:
            for line in data:
                doc_info = json.loads(line)
                doc_name = doc_info['docid']

                count += 1

                for event in doc_info['events']:
                    predicate = event['predicate']
                    frame_name = event.get('frame', None)

                    for argument in event['arguments']:
                        syn_role = argument['dep']
                        fe = argument['feName'].replace(' ', '_')

                        prop_entry = (predicate, syn_role)
                        fe_entry = (frame_name, fe)

                        if not syn_role == "NA":
                            self.arg_count[prop_entry] += 1

                        if not fe == "NA":
                            self.fe_count[fe_entry] += 1

                        if frame_name:
                            self.args_fe_map[prop_entry][fe_entry] += 1
                            self.fe_args_map[fe_entry][prop_entry] += 1

                if count % batch == 0:
                    logging.info("%d lines processed, %d errors." % (
                        count, num_format_errors))

    def write(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, 'args_frames.tsv'), 'w') as out:
            for key, count in self.arg_count.items():
                roles = ""
                if key in self.args_fe_map:
                    roles = ' '.join(
                        ['{},{}:{}'.format(p[0], p[1], c) for p, c in
                         self.args_fe_map[key].most_common()]
                    )
                out.write("%s %s\t%s\n" % (
                    ' '.join(key), count, roles
                ))

        with open(os.path.join(out_dir, 'frames_args.tsv'), 'w') as out:
            for key, count in self.fe_count.items():
                roles = ""
                if key in self.fe_args_map:
                    roles = ' '.join(
                        ['{},{}:{}'.format(p[0], p[1], c) for p, c in
                         self.fe_args_map[key].most_common()]
                    )

                out.write("%s %s\t%s\n" % (
                    ' '.join(key), count, roles
                ))

    @staticmethod
    def sorted_counts(item_counts):
        sorted_items = sorted(item_counts.items(), reverse=True,
                              key=operator.itemgetter(1))
        return ' '.join(["%s %d" % (' '.join(k), v) for k, v in sorted_items])


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    mapper = ArgFrameCollector()
    mapper.read_dataset(sys.argv[1])
    mapper.write(sys.argv[2])
