from traitlets.config import Configurable
from event.arguments.implicit_arg_params import ArgModelPara
from event.arguments.arg_models import EventCoherenceModel
from traitlets import (
    Unicode,
    Integer
)
import torch
from torch.nn import functional as F

import logging
import sys

from event.arguments.cloze_readers import HashedClozeReader

from smart_open import smart_open
from event.arguments.implicit_arg_resources import ImplicitArgResources
import math
import os
from event import torch_util
import pickle
import shutil


class ArgDecoder(Configurable):
    def __int__(self, params):
        pass

    def decode(self, coh_model, doc_event_data, doc_event_info):
        candidate_clozes = self.get_filled_clozes(doc_event_data)

        for candidate in candidate_clozes:
            scores = coh_model(candidate)

    def get_filled_clozes(self, batch_clozes):
        pass


class LocalBestDecoder(ArgDecoder):
    def __init__(self, params):
        self.frame_map = self.__load_frame_element_map()

    def __load_frame_element_map(self):
        return {}

    def decode(self, coh_model, doc_event_data, doc_event_info):
        pass

    def get_filled_list(self, doc_event_data, doc_event_info):
        # These are pre-extracted information of the events.
        mtx_event_rep = doc_event_data['rep']
        mtx_distances = doc_event_data['distances']
        mtx_features = doc_event_data['features']

        # Some fields needed to dynamically compute features.
        raw_event_info = doc_event_data['raw']

        # Context information
        mtx_context = doc_event_info['context']
        l_slots = doc_event_info['slot_indices']
        l_event_indices = doc_event_info['event_indices']


