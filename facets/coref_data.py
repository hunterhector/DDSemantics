from typing import List, Dict, Tuple, Any, Iterator

from pkg_resources import ensure_directory

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import DataPack
from forte.processors.base import PackProcessor
from forte.utils import ensure_dir
from ft.onto.base_ontology import Sentence
from transformers import AutoTokenizer

from onto.facets import Hopper, EventMention
from facets.event_utils import events2sentences, get_coref_chains, \
    all_valid_events

event_begin_marker = "<evm>"
event_end_marker = "<evm>"


def closest_pair(chain: List[EventMention]):
    pairs = []
    for i in range(len(chain)) - 1:
        pairs.append((chain[i], chain[i + 1]))
    return pairs


def pair_dataset(
        pack: DataPack,
        events2sent: Dict[int, Sentence],
        chains: List[List[int]],
) -> Iterator[Tuple[str, str, int]]:
    # Mapping from the anaphora mention index, to the chain id and its
    # position in the chain. Antecedent mentions include all mentions that are
    # not the first one of the chain.
    ana_positions = {}
    for cid, chain in enumerate(chains):
        # Every element in the chain except the first one should be anaphoric.
        for i in range(1, len(chain)):
            # Store the chain index, and the mention's relative index.
            ana_positions[chain[i]] = cid, i

    sent: Sentence
    all_events: List[EventMention] = all_valid_events(pack)

    for ana_idx in range(1, len(all_events)):
        this_evm = all_events[ana_idx]

        if this_evm.tid not in events2sent:
            import pdb
            pdb.set_trace()

        evm_in_context = bracket_mention(
            events2sent[this_evm.tid], this_evm
        )

        if ana_idx in ana_positions:
            # Take closest prior coref mention, as the chain index, and the
            # position in chain:
            cid, chain_position = ana_positions[ana_idx]
            chain = chains[cid]
            # Find the previous coref mention from the chain.
            closest_antecedent_idx = chain[chain_position - 1]
            closest_antecedent_evm = all_events[closest_antecedent_idx]

            # A positive pair.
            yield (
                bracket_mention(
                    events2sent[closest_antecedent_evm.tid],
                    closest_antecedent_evm
                ),
                evm_in_context,
                1
            )

            num_negs = 0
            if ana_idx - closest_antecedent_idx > 1:
                # Take the range in between as negative.
                for neg_idx in range(closest_antecedent_idx, ana_idx):
                    if neg_idx not in chain:
                        neg_evm = all_events[neg_idx]
                        num_negs += 1

                        yield (
                            bracket_mention(
                                events2sent[neg_evm.tid],
                                neg_evm
                            ),
                            evm_in_context,
                            0
                        )

            # Try to find a negative before the antecedent.
            if num_negs == 0:
                for idx in range(closest_antecedent_idx - 1, 0, -1):
                    if idx not in chain:
                        neg_evm = all_events[idx]

                        yield (
                            bracket_mention(
                                events2sent[neg_evm.tid],
                                neg_evm
                            ),
                            evm_in_context,
                            0
                        )
        else:
            # These are singleton mentions, so anything before it should
            # be negative. Let's take the closest one
            # TODO: This can be improved based on the mention content.
            neg_evm = all_events[ana_idx - 1]
            yield (
                bracket_mention(events2sent[neg_evm.tid], neg_evm),
                evm_in_context,
                0
            )


def bracket_mention(sent: Sentence, evm: EventMention):
    begin_offset = evm.begin - sent.begin
    end_offset = evm.end - sent.begin

    sent_text = sent.text

    bracketed = sent_text[: begin_offset] \
                + f" {event_begin_marker} " \
                + evm.text \
                + f" {event_end_marker} " \
                + sent_text[end_offset:]
    return bracketed.replace("\n", " ")


def truncate_sentence(sent, tokenizer, max_len=512):
    subwords = tokenizer(sent)
    if len(subwords) > max_len:
        print(sent)
        print(subwords)

        import pdb
        pdb.set_trace()
    else:
        return sent


class CorefInputCreator(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        ensure_dir(self.configs.output_file)
        self.output_file = open(self.configs.output_file, 'w')
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs.tokenizer)

        self._count = 0

    def _process(self, input_pack: DataPack):
        events2sents = events2sentences(input_pack)
        chains: List[List[int]] = get_coref_chains(input_pack)

        for sent1, sent2, label in pair_dataset(
                input_pack, events2sents, chains
        ):
            sent1 = truncate_sentence(sent1, self.tokenizer)
            sent2 = truncate_sentence(sent2, self.tokenizer)
            self.output_file.write(
                f"{sent1}  |||  {sent2}  |||  {label}\n"
            )
            self._count += 1

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "output_file": None,
            "tokenizer": "bert-base-cased"
        }

    def finish(self, resource: Resources):
        self.output_file.close()
        print(f"Created {self._count} instances.")
