import logging
from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
)

from nltk.corpus import (
    PropbankCorpusReader
)


class PropBank(DataLoader):
    """
    Load PropBank data.
    """

    def __init__(self, params):
        super.__init__(params)

        logging.info('Initialize PropBank reader.')


    def add_propbank_annotations(self, doc, fileid):
        pass