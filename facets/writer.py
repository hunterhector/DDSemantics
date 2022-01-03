from typing import Dict, Any

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import DataPack
from forte.processors.base import PackProcessor
from forte.utils import utils_io

from onto.facets import Hopper, EventMention


class TbfWriter(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        assert configs.output_path is not None
        assert configs.system_name is not None

        utils_io.ensure_dir(configs.output_path)
        self._tbf_out = open(configs.output_path, 'w')

    def _process(self, input_pack: DataPack):
        self._tbf_out.write(
            f"#BeginOfDocument {input_pack.pack_name}\n"
        )

        eids: Dict[int, str] = {}
        for i, evm in enumerate(input_pack.get(EventMention)):
            self._tbf_out.write(
                "\t".join(
                    [
                        self.configs.system_name,
                        input_pack.pack_name,
                        f"E{i}",
                        f"{evm.begin},{evm.end}",
                        evm.text.replace("\n", ""),
                        evm.event_type,
                        "Actual"
                    ]
                ) + "\n"
            )
            eids[evm.tid] = f"E{i}"

        hopper: Hopper
        for i, hopper in enumerate(input_pack.get(Hopper)):
            if len(hopper.get_members()) <= 1:
                continue

            member_text = ",".join(
                [eids[evm.tid] for evm in hopper.get_members()])
            self._tbf_out.write("\t".join([
                "@Coreference",
                f"R{i}",
                member_text
            ]) + "\n")

        self._tbf_out.write(
            "#EndOfDocument\n"
        )

    def finish(self, resource: Resources):
        self._tbf_out.close()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "output_path": None,
            "system_name": "empty",
        }
