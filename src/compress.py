#!/usr/bin/env python3

"""
Combines ML grayscale compression and ML colorization, using global and local color cues
"""

import os, sys
import argparse

class Compressor(object):
    def __init__(self) -> None:
        # lower CPU priority (to not freeze PC), unix only
        os.nice(10)
        pass

    def main(self):
        parser = argparse.ArgumentParser(
            prog="Compressor",
            description="Compresses and Decompresses Images using ML grayscale Compression and Colorization. ")
        parser.add_argument(
            "-d", "--decompress", action="store_true", dest="decompress", type=str,
            help="Decompress only. ")

        self.args = parser.parse_args()

if __name__ == "__main__":
    c = Compressor()
    c.main()
