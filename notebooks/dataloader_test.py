#!/usr/bin/env python
# coding: utf-8

import pathlib
import torch
import numpy as np
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
import random
from functools import partial
from copy import deepcopy
import torchaudio.transforms as T
from safetensors import safe_open
from torch import nn
from torch_audiomentations import (
    Compose,
    OneOf,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os
from time import perf_counter

SEED = 612553


def get_splits(
        data: pd.DataFrame | np.ndarray | list[...],
        train_size: float,
        valid_size: float,
        test_size: float,
        stratify_col: str | None = None,
) -> tuple[Any, Any, Any]:
    assert train_size + valid_size + test_size <= 1.0

    if stratify_col:
        train_split, valid_test = train_test_split(
            data, train_size=train_size, stratify=data[stratify_col], random_state=SEED
        )
        valid_split, test_split = train_test_split(
            valid_test, train_size=valid_size / (1 - train_size), stratify=valid_test[stratify_col], random_state=SEED
        )
    else:
        train_split, valid_test = train_test_split(data, train_size=train_size, stratify=None, random_state=SEED)
        valid_split, test_split = train_test_split(
            valid_test, train_size=valid_size / (1 - train_size), stratify=None, random_state=SEED
        )

    return train_split, valid_split, test_split

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0, p: float = 0.5) -> None:
        super().__init__()
        assert 0 <= p <= 1
        self.std: float = std
        self.mean: float = mean
        self.p: float = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn(x.size()) * self.std + self.mean if random.random() < self.p else x


class MyOneOf(torch.nn.Module):
    def __init__(self, transforms: list[nn.Module]) -> None:
        super().__init__()
        self.trasforms: list[nn.Module] = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trasforms[random.randint(0, len(self.trasforms)-1)](x)


songs_path: list[pathlib.Path] = list(pathlib.Path(os.getcwd()).parent.rglob("*.safetensors"))
train, valid, test = get_splits(songs_path, train_size=0.7, valid_size=0.2, test_size=0.1, stratify_col=None)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path: np.ndarray | list[str],
            image_size: int,
            sample_rate: int = 44100,
            crop_size: int = 60,
            mode: str = "train",
            transforms: str = "default"
    ) -> None:
        assert mode in {"train", "valid", "test"}
        super().__init__()
        self.data_path: np.ndarray | list[str] = data_path
        self.image_size: int = image_size
        self.sample_rate: int = sample_rate
        self.crop_size: int = crop_size
        self.mode: str = mode
        
        if transforms == "default":
            self._init_transforms()
        else:
            self._init_transforms_torchvision()

    def _init_transforms_torchvision(self) -> None:
        self.y_transforms = v2.Compose(
            [
                T.MelSpectrogram(
                    sample_rate=self.sample_rate, n_fft=512, win_length=512, hop_length=256, n_mels=256, normalized=True,
                ),
                v2.Resize(size=(self.image_size, self.image_size)),
                v2.ToDtype(torch.float16, scale=False),
            ]
        )

        if self.mode != "train":
            self.x_transforms = deepcopy(self.y_transforms)
            return

        self.x_transforms = v2.Compose(
            [
                AddGaussianNoise(p=0.5),
                T.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=512,
                    win_length=512,
                    hop_length=256,
                    n_mels=256,
                    normalized=True,
                ),
                MyOneOf([T.TimeMasking(time_mask_param=100), T.FrequencyMasking(freq_mask_param=100)]),
                v2.Resize(size=(self.image_size, self.image_size)),
                v2.ToDtype(torch.float16, scale=False),
            ]
        )

    def _init_transforms(self) -> None:
        # window_fn = partial(torch.hann_window(device="cuda", pin_memory=True))
        self.y_transforms = Compose(
            [
                T.MelSpectrogram(
                    sample_rate=self.sample_rate, n_fft=512, win_length=512, hop_length=256, n_mels=256, normalized=True,
                ),
                v2.Resize(size=(self.image_size, self.image_size)),
                v2.ToDtype(torch.float16, scale=False),
            ]
        )

        if self.mode != "train":
            self.x_transforms = deepcopy(self.y_transforms)
            return

        self.x_transforms = Compose(
            [
                AddGaussianNoise(p=0.5),
                T.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=512,
                    win_length=512,
                    hop_length=256,
                    n_mels=256,
                    normalized=True,
                ),
                OneOf([T.TimeMasking(time_mask_param=100), T.FrequencyMasking(freq_mask_param=100)]),
                v2.Resize(size=(self.image_size, self.image_size)),
                v2.ToDtype(torch.float16, scale=False),
            ]
        )

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # print(self.data_path[index])
        with safe_open(self.data_path[index], framework="pt", device="cpu") as f:
            sample_rate = f.get_tensor("sample_rate")
            audio = f.get_tensor("audio")

        num_frames: int = audio.shape[1]
        crop_frames: int = self.crop_size * sample_rate

        if num_frames > crop_frames:
            frame_offset: int = random.randint(0, num_frames - crop_frames)
            audio = audio[:, frame_offset : frame_offset + crop_frames]

        # audio = audio.to("cuda")
        return self.x_transforms(audio), self.y_transforms(audio)


dataloader = DataLoader(
    dataset=AudioDataset(data_path=train, image_size=256, mode="train", transforms="porcodio"),
    batch_size=8,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    persistent_workers=False,
)

for x, y in dataloader:
    break

