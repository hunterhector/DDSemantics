import logging
import os
import operator


class ArgFrameMapper:
    def __init__(self):
        self.args_frame_count = {}
        self.frame_args_count = {}
        self.len_arg_fields = 4

    def read_dataset(self, data_in):
        batch = 10000
        count = 0

        logging.info("Reading: " + data_in)

        with open(data_in) as data:
            for line in data:
                line = line.strip()
                if line.startswith("#"):
                    continue
                elif line == "":
                    continue

                count += 1

                fields = line.split("\t")

                if len(fields) < 3:
                    continue

                predicate, context, frame = fields[:3]
                arg_fields = fields[3:]
                for arg in [arg_fields[x:x + self.len_arg_fields] for x in
                            range(0, len(arg_fields), self.len_arg_fields)]:
                    prop, fe, entity, _ = arg

                    prop_entry = (predicate, prop)
                    frame_entry = (frame, fe)

                    if prop == "NA":
                        self.frame_args_count[frame_entry] = {}
                        continue

                    if fe == "NA":
                        self.args_frame_count[prop_entry] = {}
                        continue

                    from_prop = {}
                    if prop_entry in self.args_frame_count:
                        from_prop = self.args_frame_count[prop_entry]
                    else:
                        self.args_frame_count[prop_entry] = from_prop

                    try:
                        from_prop[frame_entry] += 1
                    except KeyError:
                        from_prop[frame_entry] = 1

                    from_frame = {}

                    if frame_entry in self.frame_args_count:
                        from_frame = self.frame_args_count[frame_entry]
                    else:
                        self.frame_args_count[frame_entry] = from_frame

                    try:
                        from_frame[prop_entry] += 1
                    except KeyError:
                        from_frame[prop_entry] = 1

                    if count % batch == 0:
                        logging.info("%d lines processed." % count)

    def write(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, 'args_frames.tsv'), 'w') as out:
            for key, val in self.args_frame_count.items():
                out.write("%s\t%s\n" % (
                    ' '.join(key), self.sorted_counts(val)
                ))

        with open(os.path.join(out_dir, 'frames_args.tsv'), 'w') as out:
            for key, val in self.frame_args_count.items():
                out.write("%s\t%s\n" % (
                    ' '.join(key), self.sorted_counts(val)
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
