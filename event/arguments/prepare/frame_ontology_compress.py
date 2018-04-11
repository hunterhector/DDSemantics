from nltk.corpus import framenet as fn


class FrameCompressor:
    def __init__(self):
        self.load_frame_relation()

    def load_frame_relation(self):
        pass

    def parse(self, frame_mapping_file):
        pass


if __name__ == '__main__':
    import sys

    frame_mapping_file = sys.argv[1]
    compressor = FrameCompressor()

    abuse_frame = fn.frames('Abusing')[0]

    # print(abuse_frame)
    print(abuse_frame.name)

    for relation in abuse_frame.frameRelations:
        parent_name = relation.superFrameName
        parent = fn.frames(parent_name)[0]
        print(relation.superFrameName, relation.type.name,
              relation.subFrameName)
        if relation.type.name == 'Inheritance':
            for fe in parent.FE:
                print(fe)
