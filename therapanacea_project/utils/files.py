import os

from omegaconf import OmegaConf


def make_exists(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def load_yaml(path: str):
    yaml_file = OmegaConf.load(path)
    return yaml_file
