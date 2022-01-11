"""
Module with file-readers.
"""

import json
import pickle
from pathlib import Path
from typing import Union

import yaml


def read_yml(file_name: Union[Path, str]) -> dict:
    """
    Parsing input *.yml file.

    Args:
        file_name: Name of the file with *.yml extension

    Returns:
        Content of the input *.yml file.
    """
    file_name = Path(file_name).resolve()
    with open(file_name, "r", encoding="utf-8") as file:
        yml_file = yaml.full_load(file)

    return yml_file


def read_json(file_name: Union[Path, str]) -> dict:
    """
    Parsing input *.json file.

    Args:
        file_name: Name of the file with *.json extension

    Returns:
        Content of the input *.json file.
    """
    file_name = Path(file_name).resolve()
    with open(file_name, "r", encoding='utf8') as file:
        json_file = json.load(file)

    return json_file


def save_model(file, filename) -> None:
    """
    Saving model to file.

    Args:
        file: File to be saved.
        filename: Its filename.
    """
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def load_model(filename) -> None:
    """
    Loading model from file.

    Args:
        filename: The name of model file to be loaded.

    Returns:
        The loaded model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
