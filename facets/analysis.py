from typing import Tuple, List

from forte.data import DataPack
from forte.processors.base import PackProcessor
from termcolor import colored, COLORS

from onto.facets import EventMention, Hopper


class DebugProcessor(PackProcessor):
    def _process(self, pack: DataPack):
        import pdb
        pdb.set_trace()


class EventVisualizer(PackProcessor):

    def _process(self, pack: DataPack):
        mentions = pack.get(EventMention)
        all_hoppers: List[Hopper] = list(pack.get(Hopper))

        available_colors = COLORS
        available_colors.pop("white")
        colors_to_use = list(available_colors.keys())

        if len(all_hoppers) <= len(colors_to_use):
            hopper_colors = colors_to_use[:len(all_hoppers)]
        else:
            hopper_colors = colors_to_use + ["black"] * (len(all_hoppers) - len(colors_to_use))

        coref_mention: EventMention

        span_colors = []

        for hopper, color in zip(all_hoppers, hopper_colors):
            for coref_mention in hopper.get_members():
                span_colors.append((coref_mention.begin, coref_mention.end, color))

        print(color_spans(pack.text, span_colors, bold=True))

        import pdb
        pdb.set_trace(header="Debugging this document.")


def color_spans(text: str, span_colors: List[Tuple[int, int, str]], bold=False):
    span_colors.sort(reverse=True)

    for b, e, c in span_colors:
        text = color_span(text, b, e, c, bold)

    return text


def color_span(text: str, begin: int, end: int, color: str, bold=False):
    attrs = []
    if bold:
        attrs.append("bold")
    return text[0: begin] + colored(text[begin: end], color, attrs=attrs) + text[end:]


if __name__ == '__main__':
    print(color_spans("this is a test of colors", [(5, 7, "yellow"), (0, 4, "red")]))
