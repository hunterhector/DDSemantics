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
        elif self.model_name == 'frame':
            from event.mention.models.rule_detectors import \
                FrameMappingDetector
            self.model = FrameMappingDetector(config, token_vocab)
            self.trainable = False

    def predict(self, test_reader, csr):
        for data in test_reader.read_window():
            tokens, tags, features, l_word_meta, meta = data

            event_type, args = self.model.predict(data)

            center = int(len(l_word_meta) / 2)

            token, span = l_word_meta[center]

            if not event_type == self.model.unknown_type:
                extent_span = [span[0], span[1]]
                for role, (index, entity_type) in args.items():
                    a_token, a_span = l_word_meta[index]
                    if a_span[0] < extent_span[0]:
                        extent_span[0] = a_span[0]
                    if a_span[1] > extent_span[1]:
                        extent_span[1] = a_span[1]

                evm = csr.add_event_mention(span, span, token, 'aida',
                                            event_type, component='rule')

                if evm:
                    for role, (index, entity_type) in args.items():
                        a_token, a_span = l_word_meta[index]

                        csr.add_entity_mention(a_span, a_span, a_token, 'aida',
                                               entity_type=entity_type,
                                               component='rule')

                        csr.add_event_arg_by_span(evm, a_span, a_span, a_token,
                                                  'aida', role,
                                                  component='rule')
