import json
from collections import defaultdict
import os
import sys


def get_span(frame):
    begin = frame['extent']['start']
    end = frame['extent']['length'] + begin
    return begin, end


def get_type(frame):
    return frame['interp']['type']


def construct_text(doc_sentences):
    doc_texts = {}
    for docid, content in doc_sentences.items():
        doc_text = ""
        sent_pad = ' '
        for (begin, end), sent, sid in content:
            if begin > len(doc_text):
                pad = sent_pad * (begin - len(doc_text))
                doc_text += pad
            sent_pad = '\n'
            doc_text += sent

        doc_texts[docid] = doc_text + '\n'
    return doc_texts



def strip_ns(name):
    return name.split(':', 1)[1]


def replace_ns(name):
    return '_'.join(reversed(name.split(':', 1)))


def make_text_bound(text_bound_index, t, start, end, text):
    return 'T{}\t{} {} {}\t{}\n'.format(
        text_bound_index,
        t,
        start,
        end,
        text
    )


class CrsConverter:
    def __init__(self):
        self.data = []

    def read_csr(self, path):
        with open(path) as file:
            # Better way would be convert it to CSR object.
            csr = json.load(file)

            doc_sentences = defaultdict(list)
            event_mentions = defaultdict(dict)
            event_args = defaultdict(list)
            entity_mentions = defaultdict(dict)
            relations = []

            for frame in csr['frames']:
                frame_type = frame['@type']
                if frame_type == 'document':
                    docid = frame['@id']
                    num_sentences = frame['num_sentences']
                elif frame_type == 'sentence':
                    parent = frame['parent_scope']
                    text = frame['text']
                    fid = frame['@id']
                    doc_sentences[parent].append((get_span(frame), text, fid))
                elif frame_type == 'event_mention':
                    parent = frame['parent_scope']
                    fid = frame['@id']
                    event_mentions[parent][fid] = (
                        get_span(frame), get_type(frame), frame['text'])

                    if 'args' in frame['interp']:
                        for arg in frame['interp']['args']:
                            arg_entity = arg['arg']
                            arg_type = arg['type']
                            event_args[fid].append((arg_entity, arg_type))

                elif frame_type == 'entity_mention':
                    parent = frame['parent_scope']
                    fid = frame['@id']
                    entity_mentions[parent][fid] = (
                        get_span(frame), get_type(frame), frame['text'])
                elif frame_type == 'relation_mention':
                    relations.append((get_type(frame), frame['interp']['args']))

            self.data = doc_sentences, event_mentions, event_args, entity_mentions, relations

    def write_brat(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        doc_sentences, event_mentions, event_args, entity_mentions, relations = self.data
        doc_texts = construct_text(doc_sentences)

        conf_path = os.path.join(output_dir, 'annotation.conf')

        event_types = defaultdict(dict)
        entity_types = defaultdict(set)

        for docid, doc_text in doc_texts.items():
            src_output = os.path.join(output_dir, strip_ns(docid) + '.txt')
            text_bound_index = 0
            event_index = 0
            with open(src_output, 'w') as out:
                out.write(doc_text)

            ann_output = os.path.join(output_dir, strip_ns(docid) + '.ann')

            with open(ann_output, 'w') as out:
                for sent in doc_sentences[docid]:
                    (sent_start, sent_end), sent_text, sid = sent

                    entity2tid = {}
                    for entity_id, ent in entity_mentions[sid].items():
                        span, entity_type, text = ent

                        onto, raw_type = entity_type.split(':')
                        full_type = onto + '_' + raw_type

                        if not entity_type == 'tac:arg':
                            continue

                        entity_types[onto].add(full_type)

                        text_bound = make_text_bound(
                            text_bound_index,
                            full_type,
                            sent_start + span[0], sent_start + span[1],
                            text
                        )

                        entity2tid[entity_id] = "T{}".format(text_bound_index)

                        text_bound_index += 1

                        out.write(text_bound)

                    for event_id, evm in event_mentions[sid].items():
                        span, full_type, text = evm

                        onto, raw_type = full_type.split(':')
                        full_type = onto + '_' + raw_type

                        text_bound = make_text_bound(
                            text_bound_index,
                            full_type,
                            sent_start + span[0], sent_start + span[1],
                            text
                        )

                        event_anno = 'E{}\t{}:T{}'.format(
                            event_index,
                            full_type,
                            text_bound_index
                        )

                        if full_type not in event_types[onto]:
                            event_types[onto][full_type] = set()

                        if event_id in event_args:
                            args = event_args[event_id]
                            for arg_entity, arg_type in args:
                                arg_type = replace_ns(arg_type)
                                if arg_entity in entity2tid:
                                    arg_anno = arg_type + ':' + entity2tid[
                                        arg_entity]
                                    event_anno += ' ' + arg_anno
                                    event_types[onto][full_type].add(
                                        arg_type + ':tac_arg')
                        event_anno += '\n'

                        text_bound_index += 1
                        event_index += 1

                        out.write(text_bound)
                        out.write(event_anno)

        with open(conf_path, 'w') as out:
            out.write('[entities]\n\n')

            for onto, types in entity_types.items():
                out.write('!{}\n'.format(onto + '_entity'))
                for t in types:
                    out.write('\t' + t + '\n')

            out.write('[relations]\n\n')

            out.write('[attributes]\n\n')

            out.write('[events]\n\n')
            out.write('#Definition of events.\n\n')
            out.write('other_event\trelation:tac_arg\n')

            for onto, type_args in event_types.items():
                out.write('!{}\n'.format(onto + '_event'))

                for t, args in type_args.items():
                    out.write('\t' + t)
                    sep = '\t'
                    for arg in args:
                        out.write('{}{}'.format(sep, arg))
                        sep = ', '
                    out.write('\n')


if __name__ == '__main__':
    csr_in, brat_out = sys.argv[1:3]

    converter = CrsConverter()
    converter.read_csr(csr_in)
    converter.write_brat(brat_out)
