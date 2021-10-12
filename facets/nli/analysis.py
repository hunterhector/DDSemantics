from forte.data import DataPack
from forte.processors.base import PackProcessor


class DebugProcessor(PackProcessor):
    def _process(self, pack: DataPack):
        import pdb
        pdb.set_trace()