import logging
import pathlib
import sys
from functools import lru_cache
from typing import Any

import tomllib


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


config: dict[str, Any] = get_global_config()
logger: logging.Logger = set_logger(config.get("name", ""))
