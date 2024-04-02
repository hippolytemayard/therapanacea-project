import os

from omegaconf import OmegaConf
import yaml


def make_exists(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def load_yaml(path: str):
    yaml_file = OmegaConf.load(path)
    return yaml_file


def load_yaml_(path: str):
    yaml_file = open(path, "r")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    config = OmegaConf.create(config)

    return config
