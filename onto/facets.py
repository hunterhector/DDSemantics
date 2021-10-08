# ***automatically_generated***
# ***source json:../DDSemantics/conf/nli.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology facet. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Link
from typing import Dict
from typing import Optional

__all__ = [
    "Premise",
    "Hypothesis",
    "NLIPair",
]


@dataclass
class Premise(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Hypothesis(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class NLIPair(Link):
    """
    Attributes:
        entailment (Dict[str, float]):
    """

    entailment: Dict[str, float]

    ParentType = Premise
    ChildType = Hypothesis

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.entailment: Dict[str, float] = dict()
