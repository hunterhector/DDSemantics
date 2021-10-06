from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

from forte.data import DataPack
from ft.onto.base_ontology import Sentence, Token
from ft.onto.wikipedia import WikiAnchor

stemmer = PorterStemmer()


def sentence_clues(src_sent: Sentence, src_page: str, target_pack: DataPack):
    clues = []

    tgt_sent: Sentence
    for tgt_sent in target_pack.get(Sentence):
        bidirectional = False
        for target_anchor in target_pack.get(WikiAnchor, tgt_sent):
            if target_anchor.target_page_name == src_page:
                bidirectional = True
        overlap, all_grams = compute_overlap(src_sent, tgt_sent)
        clues.append((bidirectional, overlap, tgt_sent, all_grams))
    return sorted(clues, reverse=True)


def compute_overlap(src_sent: Sentence, target_sent: Sentence):
    count = 0
    all_grams = []
    for n in (1, 2, 3):
        c, grams = ngram_overlap(src_sent, target_sent, n)
        count += c
        all_grams.extend(list(grams))
    return count, all_grams


def ngram_overlap(sent1: Sentence, sent2: Sentence, n: int):
    """
    Count the number of ngram in sent2 that appear in sent2.

    Args:
        sent1:
        sent2:
        n:

    Returns:

    """
    gram_count1 = count_gram(sent1, n)
    gram_count2 = count_gram(sent2, n)

    overlap = 0
    mapped_grams = Counter()
    for gram, count2 in gram_count2.items():
        if gram in gram_count1:
            overlap += count2
            mapped_grams[gram] += count2

    return overlap, mapped_grams


def count_gram(sent: Sentence, n: int):
    lemma_counts = Counter()

    for gram in build_ngram(sent, n):
        for g in gram:
            t = g.text.lower()
            if t not in stopwords.words(
                    "english") and t not in string.punctuation:
                lemma_counts[
                    ' '.join([stemmer.stem(g.text) for g in gram])] += 1
                break
    return lemma_counts


def build_ngram(sent: Sentence, n: int):
    # Should exclude light words from ngrams.

    if n == 1:
        return [[t] for t in sent.get(Token)]

    ngrams = []
    ngram = []
    k = 0
    for i, t in enumerate(sent.get(Token)):
        if k < n:
            ngram.append(t)
            k += 1
        else:
            if len(ngrams) > 0:
                ngram = ngram[1:] + [t]
            ngrams.append(ngram)
    return ngrams
