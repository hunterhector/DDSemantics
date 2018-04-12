import logging
import os
import operator
from collections import Counter


class ArgFrameMapper:
    def __init__(self):
        self.args_fe_map = {}
        self.fe_args_map = {}
        self.fe_count = Counter()
        self.arg_count = Counter()
        self.len_arg_fields = 5

    def read_dataset(self, data_in):
        batch = 10000
        count = 0

        logging.info("Reading: " + data_in)

        num_format_errors = 0

        with open(data_in) as data:
            for line in data:
                line = line.strip()
                if line.startswith("#"):
                    doc_name = line.rstrip("#")
                    continue
                elif line == "":
                    continue

                count += 1

                fields = line.split("\t")

                if len(fields) < 3:
                    num_format_errors += 1
                    continue

                predicate, context, frame = fields[:3]

                arg_fields = fields[3:-1]
                for arg in [arg_fields[x:x + self.len_arg_fields] for x in
                            range(0, len(arg_fields), self.len_arg_fields)]:
                    if len(arg) < self.len_arg_fields:
                        num_format_errors += 1
                        continue

                    prop, fe = arg[:2]

                    prop_entry = (predicate, prop)
                    fe_entry = (frame, fe)

                    if not prop == "NA":
                        self.arg_count[prop_entry] += 1

                    if not fe == "NA":
                        self.fe_count[fe_entry] += 1

                    # if prop == "NA":
                    #     if fe_entry not in self.fe_args_map:
                    #         self.fe_args_map[fe_entry] = {}
                    #     continue
                    #
                    # if fe == "NA":
                    #     if prop_entry not in self.args_fe_map:
                    #         self.args_fe_map[prop_entry] = {}
                    #     continue

                    from_prop = {}
                    if prop_entry in self.args_fe_map:
                        from_prop = self.args_fe_map[prop_entry]
                    else:
                        self.args_fe_map[prop_entry] = from_prop

                    try:
                        from_prop[fe_entry] += 1
                    except KeyError:
                        from_prop[fe_entry] = 1

                    from_frame = {}

                    if fe_entry in self.fe_args_map:
                        from_frame = self.fe_args_map[fe_entry]
                    else:
                        self.fe_args_map[fe_entry] = from_frame

                    try:
                        from_frame[prop_entry] += 1
                    except KeyError:
                        from_frame[prop_entry] = 1

                    if count % batch == 0:
                        logging.info("%d lines processed, %d errors." % (
                            count, num_format_errors))

    def write(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, 'args_frames.tsv'), 'w') as out:
            for key, val in self.args_fe_map.items():
                out.write("%s %s\t%s\n" % (
                    ' '.join(key), self.arg_count[key], self.sorted_counts(val)
                ))

        with open(os.path.join(out_dir, 'frames_args.tsv'), 'w') as out:
            for key, val in self.fe_args_map.items():
                out.write("%s %s\t%s\n" % (
                    ' '.join(key), self.fe_count[key], self.sorted_counts(val)
                ))

    @staticmethod
    def sorted_counts(item_counts):
        sorted_items = sorted(item_counts.items(), reverse=True,
                              key=operator.itemgetter(1))
        return ' '.join(["%s %d" % (' '.join(k), v) for k, v in sorted_items])


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    mapper = ArgFrameMapper()
    mapper.read_dataset(sys.argv[1])
    mapper.write(sys.argv[2])
