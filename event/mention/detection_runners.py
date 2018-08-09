import os


class DetectionRunner:
    def __init__(self, config, token_vocab, tag_vocab):
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.trainable = True
        self.model = None
        self.init_model(config, token_vocab, tag_vocab)

    def init_model(self, config, token_vocab, tag_vocab):
        if self.model_name == 'cnn':
            import torch
            from event.mention.models.trainable_detectors import TextCNN
            self.model = TextCNN(config, tag_vocab.vocab_size(),
                                 token_vocab.vocab_size())
            # Load model here.
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.model_name == 'frame_rule':
            from event.mention.models.rule_detectors import \
                FrameMappingDetector
            self.model = FrameMappingDetector(config, token_vocab)
            self.trainable = False
        elif self.model_name == 'marked_field':
            from event.mention.models.rule_detectors import \
                MarkedDetector
            self.model = MarkedDetector(config, token_vocab)
            self.trainable = False

    def predict(self, test_reader, csr):
        for data in test_reader.read_window():
            tokens, tags, features, l_word_meta, meta = data
            # event_types = self.model.predict(data)
            center = int(len(l_word_meta) / 2)
            token, span = l_word_meta[center]
            this_feature = features[center]

            if not this_feature[-1] == '_':
                csr.add_event_mention(span, span, token, 'aida',
                                      this_feature[-1], component='Maria')
