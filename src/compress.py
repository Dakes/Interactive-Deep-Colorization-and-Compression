#!/usr/bin/env python3

"""
Combines ML grayscale compression and ML colorization, using global and local color cues
"""

import os, sys
import argparse

import utils.files


from dinterface.preprocess import run

class Compressor(object):
    def __init__(self) -> None:
        # lower CPU priority (to not freeze PC), unix only
        os.nice(10)
        self.dirs = utils.files.config_parse(dirs=True)


    def main(self):
        parser = argparse.ArgumentParser(
            prog="Compressor",
            description="Compresses and Decompresses Images using ML grayscale Compression and Colorization. ")
        parser.add_argument(
            "-d", "--decompress", action="store_true", dest="decompress", type=str,
            help="Decompress only. ")

        self.args = parser.parse_args()



    def theme_extract(self):
        pass


if __name__ == "__main__":
    c = Compressor()
    c.main()
