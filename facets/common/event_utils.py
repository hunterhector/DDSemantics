from typing import Dict, List

from forte.data import DataPack
from ft.onto.base_ontology import Sentence

from onto.facets import EventArgument, EventMention, EntityMention, Hopper


def events2sentences(pack: DataPack) -> Dict[int, Sentence]:
    events2sents: Dict[int, Sentence] = {}
    for sent in pack.get(Sentence):
        for evm in sent.get(EventMention):
            events2sents[evm.tid] = sent
    return events2sents


def all_valid_events(pack: DataPack) -> List[EventMention]:
    """
    Some events are not in filtered text. We ignore them.

    Args:
        pack:

    Returns:

    """
    all_events: List[EventMention] = []
    for sent in pack.get(Sentence):
        all_events.extend(sent.get(EventMention))
    return all_events


def get_coref_chains(pack: DataPack) -> List[List[int]]:
    """

    Args:
        pack:

    Returns: Coref chains, where each chain is the indices of the mention.

    """
    evm_id2index = {}

    for idx, mention in enumerate(all_valid_events(pack)):
        evm_id2index[mention.tid] = idx

    chains: List[List[int]] = []

    hopper: Hopper
    for hopper in pack.get(Hopper):
        chain = []
        for mention in hopper.get_members():
            # Invalid mentions should be removed.
            if mention.tid in evm_id2index:
                idx = evm_id2index[mention.tid]
                chain.append(idx)
        if len(chain) > 1:
            chains.append(sorted(chain))
    return chains


def build_arguments(pack: DataPack):
    all_args: Dict[int, Dict[str, int]] = {}

    argument: EventArgument
    for argument in pack.get(EventArgument):
        evm: EventMention = argument.get_parent()
        arg: EntityMention = argument.get_child()

        try:
            all_args[evm.tid][argument.role] = arg
        except KeyError:
            all_args[evm.tid] = {argument.role: arg}

    return all_args
