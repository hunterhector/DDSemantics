# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/DDSemantics/conf/facet.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology facet. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics
from forte.data.ontology.top import Group
from forte.data.ontology.top import Link
from forte.data.ontology.top import MultiPackLink
from ft.onto.base_ontology import EntityMention as EntityMention_0
from ft.onto.base_ontology import EventMention as EventMention_0
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

__all__ = [
    "Premise",
    "Hypothesis",
    "NLIPair",
    "EntityMention",
    "EventMention",
    "EventArgument",
    "Hopper",
    "Facet",
    "CopyLink",
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
class EntityMention(EntityMention_0):
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
class EventMention(EventMention_0):
    """
    Attributes:
        types (List[str]):
        realis (Optional[str]):
        audience (Optional[str]):
        formality (Optional[str]):
        medium (Optional[str]):
        schedule (Optional[str]):
        id (Optional[str]):
    """

    types: List[str]
    realis: Optional[str]
    audience: Optional[str]
    formality: Optional[str]
    medium: Optional[str]
    schedule: Optional[str]
    id: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.types: List[str] = []
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
        pb_role (Optional[str]):
        vb_role (Optional[str]):
        realis (Optional[str]):
        id (Optional[str]):
    """

    role: Optional[str]
    pb_role: Optional[str]
    vb_role: Optional[str]
    realis: Optional[str]
    id: Optional[str]

    ParentType = EventMention
    ChildType = EntityMention

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.role: Optional[str] = None
        self.pb_role: Optional[str] = None
        self.vb_role: Optional[str] = None
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


@dataclass
class Facet(Generics):
    """
    Attributes:
        facet_name (Optional[str]):
    """

    facet_name: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.facet_name: Optional[str] = None


@dataclass
class CopyLink(MultiPackLink):
    """
    indicate that the child entry is copied from the parent entry
    """

    ParentType = Entry
    ChildType = Entry

    def __init__(self, pack: MultiPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
