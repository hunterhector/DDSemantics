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
from forte.data.ontology.top import Group
from forte.data.ontology.top import Link
from ft.onto.base_ontology import EntityMention
from ft.onto.base_ontology import EventMention
from typing import Dict
from typing import Iterable
from typing import Optional

__all__ = [
    "Premise",
    "Hypothesis",
    "NLIPair",
    "EntityMention",
    "EventMention",
    "EventArgument",
    "Hopper",
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


@dataclass
class EntityMention(EntityMention):
    """
    Attributes:
        id (Optional[str]):
        is_filler (Optional[bool]):
    """

    id: Optional[str]
    is_filler: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.id: Optional[str] = None
        self.is_filler: Optional[bool] = None


@dataclass
class EventMention(EventMention):
    """
    Attributes:
        realis (Optional[str]):
        audience (Optional[str]):
        formality (Optional[str]):
        medium (Optional[str]):
        schedule (Optional[str]):
        id (Optional[str]):
    """

    realis: Optional[str]
    audience: Optional[str]
    formality: Optional[str]
    medium: Optional[str]
    schedule: Optional[str]
    id: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.realis: Optional[str] = None
        self.audience: Optional[str] = None
        self.formality: Optional[str] = None
        self.medium: Optional[str] = None
        self.schedule: Optional[str] = None
        self.id: Optional[str] = None


@dataclass
class EventArgument(Link):
    """
    Attributes:
        role (Optional[str]):
        realis (Optional[str]):
        id (Optional[str]):
    """

    role: Optional[str]
    realis: Optional[str]
    id: Optional[str]

    ParentType = EventMention
    ChildType = EntityMention

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.role: Optional[str] = None
        self.realis: Optional[str] = None
        self.id: Optional[str] = None


@dataclass
class Hopper(Group):
    """
    Attributes:
        id (Optional[str]):
    """

    id: Optional[str]

    MemberType = EventMention

    def __init__(self, pack: DataPack, members: Optional[Iterable[Entry]] = None):
        super().__init__(pack, members)
        self.id: Optional[str] = None
