import dataclasses
import logging
import pathlib
import sys
from functools import lru_cache
from http import HTTPStatus
from typing import Type

import requests
import tomllib
from cachetools.func import ttl_cache
from torch import nn


@dataclasses.dataclass
class Hyperparameters:
    """Data class for storing hyperparameters used in training.

    Attributes:
        BATCH_SIZE (int): The batch size for training.
        INPUT_SIZE (int): The size of the input tensor.
        EPOCHS (int): The number of training epochs.
        LEARNING_RATE (float): The learning rate for the optimizer.
        LEARNING_RATE_DECAY (int): The learning rate decay.
        TRAIN_SIZE (float): The proportion of the data used for training.
        BASE_CHANNEL_SIZE (int): The base number of channels for the model.
        LATENT_DIM (int): The dimension of the latent space.
        NUM_INPUT_CHANNELS (int): The number of input channels.
        T_0 (int): The T_0 parameter for the learning rate scheduler.
        T_mult (int): The T_mult parameter for the learning rate scheduler.
        EARLY_STOPPING_PATIENCE (int): The patience for early stopping.
        EARLY_STOPPING_MIN_DELTA (float): The minimum delta for early stopping.
        OVERFIT_BATCHES (int): The number of batches to overfit on.
        LOSS (nn.Module): The loss function used for training.
        CROP_SIZE_SECONDS (int): The size of the cropped audio in seconds.
        SEED (int): The random seed for reproducibility.
        PRECISION (int): The floating point precision used for training.
        GRADIENT_ACCUMULATION_BATCHES (int): The number of batches to run before updating the weights
    """

    BATCH_SIZE: int
    INPUT_SIZE: int
    TRAIN_SIZE: float
    VALID_SIZE: float
    TEST_SIZE: float
    BASE_CHANNEL_SIZE: int
    LATENT_DIM: int
    NUM_INPUT_CHANNELS: int
    EPOCHS: int
    LEARNING_RATE: float
    LEARNING_RATE_DECAY: float
    T_0: int
    T_mult: int
    EARLY_STOPPING_PATIENCE: int
    EARLY_STOPPING_MIN_DELTA: float
    OVERFIT_BATCHES: int
    CROP_SIZE_SECONDS: int
    SEED: int
    GRADIENT_CLIP_VAL: float
    GRADIENT_CLIP_TYPE: str
    PRECISION: int
    GRADIENT_ACCUMULATION_BATCHES: int
    LOSS: nn.Module = nn.L1Loss
    N_FFT: int = 512
    WIN_LENGTH: int = 512
    HOP_LENGTH: int = 256
    N_MELS: int = 256


def get_global_config() -> dict[str, str | int | float | bool]:
    """Return the global configuration from the pyproject.toml file.

    Returns:
        dict[str, str | int | float | bool]: The global configuration.

    Examples:
        >>> config = get_global_config()
    """
    a = pathlib.Path.cwd()
    with pathlib.Path("pyproject.toml").open("rb") as f:
        return tomllib.load(f)


@lru_cache()
def set_logger(name: str) -> logging.Logger:
    """Set up and return a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.

    Examples:
        >>> logger = set_logger("my_logger")
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


@ttl_cache(maxsize=1, ttl=3600)
def get_spotify_access_token() -> str:
    """Return the Spotify access token.

    Returns:
        str: The Spotify access token.

    Examples:
        >>> access_token = get_spotify_access_token()
    """
    response: requests.Response = requests.post(
        url="https://accounts.spotify.com/api/token",
        data="grant_type=client_credentials&client_id=bf621646332d4c9c82c6e6d1fd8a8352&client_secret"
        "=0ecda4e3308e4340a26b519d0647b2bf",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return response.json().get("access_token") if response.status_code == HTTPStatus.OK else ""


def get_dir_absolute_path(dir_name: str) -> pathlib.Path:
    """Return the absolute path of the directory with the specified name.

    It searches in all subdirectories of the cwd parent folder and returns the absolute path of the directory named
    `dir_name`
    Args:
        dir_name (str): The name of the directory.

    Returns:
        pathlib.Path: The absolute path of the directory.

    Examples:
        >>> dir_path = get_dir_absolute_path("my_directory")
    """
    current_folder: pathlib.Path = pathlib.Path.cwd()

    target_folder_path: pathlib.Path = pathlib.Path()
    for parent in current_folder.parents:
        for potential_folder_path in parent.rglob(dir_name):
            if potential_folder_path.is_dir():
                return potential_folder_path

    return target_folder_path


# def move_songs_to_release_date_partition(song: dict[str, Any]) -> None:
#     songs_path: pathlib.Path = get_dir_absolute_path("songs")
#     original_song_path: pathlib.Path = (songs_path / song.get("track", {})["id"]).with_suffix(".mp3")
#
#     if not original_song_path.exists():
#         return
#
#     try:
#         release_path: pathlib.Path = get_song_realease_date_folder(
#             release_date=song.get("track", {})["album"]["release_date"],
#             precision=song.get("track", {})["album"]["release_date_precision"],
#         )
#         output_path: pathlib.Path = (songs_path / release_path / song.get("track", {})["id"]).with_suffix(".mp3")
#     except Exception:
#         return
#
#     if output_path.exists():
#         return
#
#     logger.debug('moving song.get("track", {})["id"]')
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     shutil.move(original_song_path, output_path)

config: dict[str, str | float | int | bool] = get_global_config()
logger: logging.Logger = set_logger(config.get("title", ""))


def dataclass_from_dict(class_: Type, dictionary: dict[str, str | float | int]) -> dict[str, str | float | int] | Type:
    """Converts a dictionary to a dataclass instance.

    Args:
        class_: The dataclass type.
        dictionary: The dictionary to convert.

    Returns:
        Union[dict[str, Union[str, float, int]], dataclasses.dataclass]: The converted dataclass instance or the
            original dictionary.
    """
    try:
        field_types: dict = {f.name: f.type for f in dataclasses.fields(class_)}
        # print(f"field_types: {field_types}")
        # print(f"dictionary: {dictionary}")
        return class_(**{f: dataclass_from_dict(field_types.get(f), dictionary.get(f)) for f in dictionary})
    except Exception:
        # print(f"EXCEPTION: {e}")
        return dictionary  # The object is not a dataclass field
