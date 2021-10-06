import datetime
import sys
import timeit

from forte.common import Resources
from forte.common.configuration import Config
from forte.data import DataPack
from forte.processors.base import PackProcessor


class ProgressPrinter(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.count = 0
        self.start_time = timeit.default_timer()
        print("Initialization Done.")

    def _process(self, input_pack: DataPack):
        self.count += 1
        elapsed = timeit.default_timer() - self.start_time
        print_progress(
            f"Handling the pack [{input_pack.pack_name}] ({self.count}) after "
            f"{datetime.timedelta(seconds=elapsed)}, speed is "
            f"{self.count / elapsed:.2f} (packs/second)."
        )

    def finish(self, resource: Resources):
        print("\nProgress Printer Finished.")

def print_progress(msg: str, end="\r"):
    sys.stdout.write("\033[K")  # Clear to the end of line.
    print(f" -- {msg}", end=end)
