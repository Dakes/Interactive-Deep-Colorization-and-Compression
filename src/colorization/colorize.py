#!/usr/bin/env python3

"""
Extracts color theme and individual pixels, for best recolorization.
Saves them and can recolor grayscale images using these color hints. (Also grayscale compressed ones)

"""

import os, sys
import argparse

class Colorizer(object):
    def __init__(self, recolorize, compress) -> None:
        self.recolorize = recolorize
        self.compress = compress
        # lower CPU priority (to not freeze PC), unix only
        os.nice(10)
        pass

    def main(self):
        parser = argparse.ArgumentParser(
            prog="Colorizer",
            description="Recolorizes images using Xiao et al's  colorization system")
        parser.add_argument(
            "-r", "--recolorize", action="store_true", dest="decompress", type=str,
            help="Recolorize only. ")
        parser.add_argument(
            "-c", "--compress", action="store_true", dest="compress", type=str,
            help="Compress (save color cues) only. ")

        self.args = parser.parse_args()
        self.recolorize = self.args.recolorize
        self.compress = self.args.compress

    def color_cue_gen(self):
        pass

    def recolorize(self):
        pass



if __name__ == "__main__":
    c = Colorizer()
    c.main()
