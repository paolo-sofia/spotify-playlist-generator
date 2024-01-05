import pathlib

import tomllib
import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from torchvision.transforms import v2

from src.db.schemas.song_embedding import SongEmbedding
from src.model.autoencoder import Autoencoder
from src.model.dataset.transforms import MinMaxNorm
from src.model.hyperparameters import Hyperparameters
from src.utils.utils import dataclass_from_dict


def define_transforms(config: Hyperparameters) -> v2.Compose:
    """Define the data transformations for audio input based on the provided configuration.

    Args:
        config: The hyperparameter configuration.

    Returns:
        v2.Compose: The Compose object containing data transformations.
    """
    precision: torch.dtype = {16: torch.float16, 32: torch.float32, 64: torch.float64}.get(config.PRECISION, 32)

    return v2.Compose(
        [
            T.MelSpectrogram(
                sample_rate=config.SAMPLE_RATE,
                n_fft=config.N_FFT,
                win_length=config.WIN_LENGTH,
                hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS,
                normalized=False,
            ),
            MinMaxNorm(),
            v2.ToDtype(precision, scale=False),
        ]
    )


def preprocess(tensor: torch.Tensor) -> torch.Tensor:
    return transforms(tensor).to(device).unsqueeze(dim=0)


def load_model() -> nn.Module:
    return Autoencoder.load_from_checkpoint("", hyperparam=cfg).eval().half().to(device)


def load_model_hyperparameters() -> Hyperparameters:
    with (pathlib.Path.cwd() / "config" / "model.toml").open("rb") as f:
        return dataclass_from_dict(Hyperparameters, tomllib.load(f))


def wrap_prediction_to_song_embedding(tensor: torch.Tensor, song_id: str) -> SongEmbedding:
    print(tensor.shape)
    return SongEmbedding(id=song_id, embedding=tensor.tolist())


def predict_audio(audio: torch.Tensor) -> torch.Tensor:
    num_frames: int = audio.shape[1]
    last_audio_slice_length: int = num_frames % cfg.CROP_FRAMES

    if last_audio_slice_length == 0:
        pass
    elif last_audio_slice_length > cfg.CROP_FRAMES / 2:
        frames_to_add: int = cfg.CROP_FRAMES - (num_frames % cfg.CROP_FRAMES)
        audio = torch.cat([audio, torch.zeros((audio.shape[0], frames_to_add))], dim=1)
    else:
        frames_to_remove: int = num_frames % cfg.CROP_FRAMES
        audio = audio[:, : num_frames - frames_to_remove]

    num_slices: int = audio.shape[1] // cfg.CROP_FRAMES
    print(f"num_frames: {audio.shape[1]} - num_slices: {num_slices}")
    slices: list[torch.Tensor] = torch.chunk(audio, num_slices, dim=1)
    return torch.stack([model.encoder(preprocess(audio_slice)) for audio_slice in slices]).squeeze().mean(dim=0)


def get_song_embedding(track: dict[str, str]) -> SongEmbedding:
    song_id: str = track.get("song_id")
    audio_path: str = track.get("audio_path")

    audio, _ = torchaudio.load(audio_path)
    num_frames: int = audio.shape[1]
    if num_frames <= cfg.CROP_FRAMES:
        frames_to_add: int = cfg.CROP_FRAMES * (num_frames // cfg.CROP_FRAMES + 1) - num_frames
        audio: torch.Tensor = torch.cat(tensors=[audio, torch.zeros((audio.shape[0], frames_to_add))], dim=1)

    return wrap_prediction_to_song_embedding(predict_audio(audio), song_id)


cfg: Hyperparameters = load_model_hyperparameters()
model: nn.Module = load_model()
transforms: v2.Compose = define_transforms(cfg)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "")
