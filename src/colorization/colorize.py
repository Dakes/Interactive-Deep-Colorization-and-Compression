#!/usr/bin/env python3

"""
Extracts color theme and individual pixels, for best recolorization.
Saves them and can recolor grayscale images using these color hints. (Also grayscale compressed ones)

"""

import os, sys
import argparse

import src.dinterface.utils
import src.dinterface.preprocess
import src.utils.files as files

class Colorizer(object):
    def __init__(self, recolorize, compress, num_points=6) -> None:
        self.recolorize = recolorize
        self.compress = compress
        self.num_points = num_points
        self.dirs = files.config_parse(dirs=True)
        # lower CPU priority (to not freeze PC), unix only
        os.nice(10)

        self.num_points_pix = -1
        self.num_points_theme = 6
        # for training (set to False to not crop)
        self.random_crop = 256


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



    def color_cue_gen(self, rgb_img):
        # TODO: maybe implement single image mode
        src.dinterface.utils.preprocess(self.dirs)


    def recolorize(self):
        pass



if __name__ == "__main__":
    c = Colorizer()
    c.main()
