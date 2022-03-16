"""
File and path related utils and helper functions
"""

import os, sys
import configparser
from pathlib import Path


def create_dirs(dirs):
    # create dir if not exists
    for key in dirs:
        if not os.path.exists(dirs[key]):
            os.makedirs(dirs[key])

def config_parse(dirs=False):
    """
    :param dirs: return dirs section only
    returns parsed config from src/config.cfg
    """
    proj_root = Path(os.path.dirname(os.path.realpath(__file__))).parent
    config_path = os.path.join(proj_root, "config.cfg")
    config = configparser.ConfigParser()
    config.read(config_path)
    create_dirs(config["dir"])
    if dirs:
        return config["dir"]
    else:
        return config


