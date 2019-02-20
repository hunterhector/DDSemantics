import os
from collections import defaultdict

from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
)


class Conll(DataLoader):
    def __init__(self, params, with_doc=False):
        super().__init__(params, with_doc)
        self.params = params

    def parse_conll_data(self, corpus, conll_in):
        text = ''
        offset = 0

        arg_text = []
        sent_predicates = []
        sent_args = defaultdict(list)
        doc = DEDocument(corpus)

        props = []

        for line in conll_in:
            parts = line.strip().split()
            if len(parts) < 8:
                text += '\n'
                offset += 1

                for index, predicate in enumerate(sent_predicates):
                    arg_content = sent_args[index]
                    props.append((predicate, arg_content))

                sent_predicates.clear()
                sent_args.clear()
                arg_text.clear()

                continue

            fname, _, index, token, pos, parse, lemma, sense = parts[:8]
            pb_annos = parts[8:]

            if len(arg_text) == 0:
                arg_text = [None] * len(pb_annos)

            domain = fname.split('/')[1]

            start = offset
            end = start + len(token)

            text += token + ' '
            offset += len(token) + 1

            for index, t in enumerate(arg_text):
                if t:
                    arg_text[index] += ' ' + token

            if not sense == '-':
                sent_predicates.append((start, end, token))

            for index, anno in enumerate(pb_annos):
                if anno == '(V*)':
                    continue

                if anno.startswith('('):
                    role = anno.strip('(').strip(')').strip('*')
                    sent_args[index].append([role, start])
                    arg_text[index] = token
                if anno.endswith(')'):
                    sent_args[index][-1].append(end)
                    sent_args[index][-1].append(arg_text[index])
                    arg_text[index] = ''

        doc.set_text(text)

        for (p_start, p_end, p_token), args in props:
            hopper = doc.add_hopper()

            pred = doc.add_predicate(
                hopper, Span(p_start, p_end), p_token)

            if pred is not None:
                for role, arg_start, arg_end, arg_text in args:
                    filler = doc.add_filler(Span(arg_start, arg_end), arg_text)
                    doc.add_argument_mention(pred, filler.aid, role)

        return doc

    def get_doc(self):
        super().get_doc()
        for dirname in os.listdir(self.params.in_dir):
            full_dir = os.path.join(self.params.in_dir, dirname)
            for root, dirs, files in os.walk(full_dir):
                for f in files:
                    if not f.endswith('gold_conll'):
                        continue

                    full_path = os.path.join(root, f)

                    out_dir = os.path.join(self.params.out_dir, dirname)

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    docid = f.replace('gold_conll', '')

                    with open(full_path) as conll_in:
                        doc = self.parse_conll_data(self.corpus, conll_in)
                        doc.set_id(docid)
                        yield doc
