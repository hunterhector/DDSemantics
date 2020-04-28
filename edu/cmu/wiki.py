# ***automatically_generated***
# ***source json:../DDSemantics/wikipedia.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology wikipedia. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import Optional


__all__ = [
    "PredictedWikiAnchor",
]


class PredictedWikiAnchor(Annotation):
    """
    Attributes:
        _target_page_name (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._target_page_name: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['target_page_name'] = state.pop('_target_page_name')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._target_page_name = state.get('target_page_name', None) 

    @property
    def target_page_name(self):
        return self._target_page_name

    @target_page_name.setter
    def target_page_name(self, target_page_name: Optional[str]):
        self.set_fields(_target_page_name=target_page_name)
