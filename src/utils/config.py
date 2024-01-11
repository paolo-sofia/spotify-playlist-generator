import logging
import pathlib
import sys
from functools import lru_cache
from typing import Any

import tomllib

from src.model.hyperparameters import Hyperparameters
from src.utils.utils import dataclass_from_dict


@lru_cache()
def set_logger(name: str) -> logging.Logger:
    """Set up and return a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.

    Examples:
        >>> logger_ = set_logger("my_logger")
    """
    log: logging.Logger = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))
    return log


def get_global_config() -> dict[str, str | int | float | bool]:
    """Return the global configuration from the pyproject.toml file.

    Returns:
        dict[str, str | int | float | bool]: The global configuration.

    Examples:
        >>> config_ = get_global_config()
    """
    try:
        with pathlib.Path("config/project.toml").open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def load_model_hyperparameters() -> Hyperparameters:
    with (pathlib.Path.cwd() / "config" / "model.toml").open("rb") as f:
        return dataclass_from_dict(Hyperparameters, tomllib.load(f))


config: dict[str, Any] = get_global_config()
logger: logging.Logger = set_logger(config.get("name", ""))
