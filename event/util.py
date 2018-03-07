import argparse
import logging
import sys


class OptionPerLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []
        return arg_line.split()


def set_basic_log(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
