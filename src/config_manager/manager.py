"""Configuration manager"""

from typing import Dict, Union

import yaml


class Params:
    """Class to load hyperparameters from a yaml file."""

    def __init__(self, inp: Union[Dict, str]) -> None:
        self.update(inp)

    def save(self, yaml_path: str) -> None:
        """Save parameters to yaml file at yaml_path"""
        with open(yaml_path, "w", encoding="utf-8") as fptr:
            yaml.safe_dump(self.__dict__, fptr)

    def update(self, inp: Union[Dict, str]) -> None:
        """Loads parameters from yaml file or dict"""
        if isinstance(inp, dict):
            self.__dict__.update(inp)
        elif isinstance(inp, str):
            with open(inp, encoding="utf-8") as fptr:
                params = yaml.safe_load(fptr)
                self.__dict__.update(params)
        else:
            raise TypeError(
                "Input should either be a dictionary or a string path to a config file!"
            )

    def __str__(self) -> str:
        """Print instance"""
        return str(self.__dict__)
