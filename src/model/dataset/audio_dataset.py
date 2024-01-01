import random
from copy import deepcopy
from typing import Self

import numpy as np
import torch
import torchaudio.transforms as T
from safetensors import safe_open
from torch import nn
from torch_audiomentations import OneOf
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

from .transforms import AddGaussianNoise, MinMaxNorm


class AudioDataset(torch.utils.data.Dataset):
    """Dataset class for audio data.

    Args:
        data_path: The path to the audio data.
        image_size: The desired size of the audio spectrogram images.
        crop_size: The size of the cropped audio in seconds.
        mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
    """

    def __init__(
        self: Self,
        data_path: np.ndarray | list[str],
        image_size: int,
        crop_size: int,
        precision: int,
        mode: str = "train",
    ) -> None:
        """Initializes the AudioDataset with the specified parameters.

        Args:
            data_path: The path to the audio data.
            image_size: The desired size of the audio spectrogram images.
            crop_size: The size of the cropped audio in seconds.
            mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
        """
        assert mode in {"train", "valid", "test"}
        super().__init__()
        self.data_path: np.ndarray | list[str] = data_path
        self.image_size: int = image_size
        self.crop_size: int = crop_size
        self.mode: str = mode
        self.precision: dict[int, torch.float] = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }.get(precision)
        # self._init_transforms()

    def _get_transforms(self: Self, sample_rate: int) -> tuple[Compose, Compose]:
        transforms: list[nn.Module] = [
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                win_length=512,
                hop_length=256,
                n_mels=256,
                normalized=False,
            ),
            MinMaxNorm(),
            v2.Resize(size=(self.image_size, self.image_size)),
            v2.ToDtype(self.precision, scale=False),
        ]

        y_transforms = Compose(transforms)
        train_transforms = deepcopy(transforms)

        if self.mode == "train":
            train_transforms.insert(1, AddGaussianNoise(p=1.0))
            train_transforms.insert(
                2, OneOf([T.TimeMasking(time_mask_param=100), T.FrequencyMasking(freq_mask_param=30)])
            )

        return Compose(train_transforms), y_transforms

    def __len__(self: Self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_path)

    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the item at the specified index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
        # print(self.data_path[index])
        with safe_open(self.data_path[index], framework="pt", device="cpu") as f:
            sample_rate: int = f.get_tensor("sample_rate")
            audio: torch.Tensor = f.get_tensor("audio")

        num_frames: int = audio.shape[1]
        crop_frames: int = self.crop_size * sample_rate
        x_transforms, y_transforms = self._get_transforms(sample_rate)
        if num_frames <= crop_frames:
            return x_transforms(audio), y_transforms(audio)

        x_transformed: torch.Tensor
        y_transformed: torch.Tensor
        while True:
            frame_offset: int = random.randint(0, num_frames - crop_frames)
            cropped_audio = audio[:, frame_offset : frame_offset + crop_frames]
            x_transformed, y_transformed = x_transforms(cropped_audio), y_transforms(cropped_audio)
            if not torch.isnan(x_transformed).sum() and not torch.isnan(x_transformed).sum():
                break
        return x_transformed, y_transformed
