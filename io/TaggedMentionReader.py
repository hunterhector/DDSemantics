import os
from nltk import tokenize
import nltk
import string
from collections import defaultdict
import glob


class TaggedMentionReader:
    def __init__(self, config):
        self.data_folder = config['data_folder']

        self.tag_folders = os.path.join(self.data_folder, "tags")
        self.source_folders = os.path.join(self.data_folder, "sources")

        self.data_configs = config['data_configs']

        self.no_punct = config['no_punct']

        self.token2i = defaultdict(lambda: len(self.token2i))
        self.tag2i = defaultdict(lambda: len(self.tag2i))

        self.unk = self.token2i["<unk>"]
        self.none_tag = self.tag2i["NONE"]

        tokenizer_type = config['tokenizer_type']
        language = config['language']
        if tokenizer_type == 'twitter':
            self.tokenizer = tokenize.TweetTokenizer
        elif tokenizer_type == 'moses':
            self.tokenizer = tokenize.moses.MosesDetokenizer(lang=language)
        else:
            # Default word tokenzier.
            self.tokenizer = tokenize.treebank.TreebankWordTokenizer()

    def hash(self):
        import pickle
        tag_path = os.path.join(self.data_folder, 'tags_dict.pickle')
        source_path = os.path.join(self.data_folder, 'tokens_dict.pickle')

        if os.path.exists(tag_path) and os.path.exists(source_path):
            with open(tag_path) as tag_pickle:
                self.tag2i = pickle.load(tag_pickle)
            with open(source_path) as source_pickle:
                self.token2i = pickle.load(source_pickle)
        else:
            # Hash the tokens and tags.
            for tag_folder, source_folder, data_config in zip(
                    self.tag_folders, self.source_folders, self.data_configs):
                text_suffix = data_config['text_suffix']
                tag_suffix = data_config['tag_suffix']

                for tag_pickle in glob.glob(tag_folder + "/*." + tag_suffix):
                    docid = os.path.basename(tag_pickle).replace(tag_suffix, "")

                    source_file = os.path.join(source_folder,
                                               docid + text_suffix)

                    with open(source_file) as source:
                        for sent in source:
                            tokens = self.tokenizer.tokenize(sent)
                            [self.token2i[t] for t in tokens]

                    with open(tag_pickle) as tag_in:
                        for line in tag_in:
                            tags = line.split(" ")
                            [self.tag2i[t] for t in tags]

            self.token2i = defaultdict(lambda: self.unk, self.token2i)
            self.tag2i = defaultdict(lambda: self.none_tag, self.tag2i)

    def read(self):
        for tag_folder, source_folder, data_config in zip(self.tag_folders,
                                                          self.source_folders,
                                                          self.data_configs):
            text_suffix = data_config['text_suffix']
            tag_suffix = data_config['tag_suffix']

            for tag_file in glob.glob(tag_folder + "/*." + tag_suffix):
                docid = os.path.basename(tag_file).replace(tag_suffix, "")

                source_file = os.path.join(source_folder, docid + text_suffix)

                with open(source_file) as source:
                    for sent in source:
                        tokens = self.tokenizer.tokenize(sent)
                        token_ids = [self.token2i[t] for t in tokens]

                with open(tag_file) as tag_in:
                    for line in tag_in:
                        tags = line.split(" ")
                        tag_ids = [self.tag2i[t] for t in tags]

                yield token_ids, tag_ids
