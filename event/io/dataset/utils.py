import logging
from nltk.corpus.reader.nombank import (
    NombankChainTreePointer,
    NombankSplitTreePointer,
    NombankTreePointer,
)
from nltk.corpus.reader.propbank import (
    PropbankChainTreePointer,
    PropbankSplitTreePointer
)
from event.io.dataset.base import Span


def get_tree_pointers(tree_pointer):
    pointers = []
    if isinstance(tree_pointer, NombankSplitTreePointer) or isinstance(
            tree_pointer, NombankChainTreePointer):
        for p in tree_pointer.pieces:
            pointers.extend(get_tree_pointers(p))
    elif isinstance(tree_pointer, PropbankSplitTreePointer) or isinstance(
            tree_pointer, PropbankChainTreePointer):
        for p in tree_pointer.pieces:
            pointers.extend(get_tree_pointers(p))
    else:
        pointers.append(tree_pointer)
    return sorted(pointers, key=lambda pt: pt.wordnum)


def make_words_from_pointer(tree, tree_pointer):
    """
    Create words from tree pointer (NLTK).
    :param tree: The tree for the whole sentence.
    :param tree_pointer: The tree pointer that point to some nodes.
    :return:
    """
    pointers = get_tree_pointers(tree_pointer)

    all_word_idx = []
    all_word_surface = []

    for pointer in pointers:
        treepos = pointer.treepos(tree)

        idx_list = []
        for idx in range(len(tree.leaves())):
            if tree.leaf_treeposition(idx)[:len(treepos)] == treepos:
                idx_list.append(idx)

        idx_list.sort()
        word_list = [tree.leaves()[idx] for idx in idx_list]

        if len(all_word_idx) > 0 and \
                idx_list[0] - 1 == all_word_idx[-1][-1]:
            all_word_idx[-1].extend(idx_list)
            all_word_surface[-1].extend(word_list)
        else:
            all_word_idx.append(idx_list)
            all_word_surface.append(word_list)

    return all_word_idx


def get_nltk_span(token_spans, sent_num, indice_groups):
    spans = []
    for indices in indice_groups:
        start = -1
        end = -1
        for index in indices:
            s = token_spans[sent_num][index]
            if s:
                if start < 0:
                    start = s[0]
                end = s[1]

        if start >= 0 and end >= 0:
            spans.append(Span(start, end))
    return spans


def nombank_pred_text(raw_text):
    p_text = raw_text.lower()

    if p_text.startswith('not_'):
        p_text = p_text[4:]

    if p_text == 'losses' or p_text == 'loss' or p_text == 'tax-loss':
        p_text = 'loss'
    else:
        p_text = p_text.rstrip('s')

    if p_text == 'savings-and-loan':
        p_text = 'loan'

    if '-' in p_text:
        p_text = p_text.split('-')[1]
    return p_text
