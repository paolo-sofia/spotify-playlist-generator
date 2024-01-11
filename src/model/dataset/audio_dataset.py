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

from .transforms import MinMaxNorm


class AudioDataset(torch.utils.data.Dataset):
    """Dataset class for audio data.

    Args:
        data_path: The path to the audio data.
        crop_size: The size of the cropped audio in seconds.
        mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
    """

    def __init__(
        self: Self,
        data_path: np.ndarray | list[str],
        crop_size: int,
        precision: int,
        mel_spectrogram_param: dict[str, int],
        mode: str = "train",
    ) -> None:
        """Initializes the AudioDataset with the specified parameters.

        Args:
            data_path: The path to the audio data.
            crop_size: The size of the cropped audio in seconds.
            mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
        """
        assert mode in {"train", "valid", "test"}
        super().__init__()
        self.data_path: np.ndarray | list[str] = data_path
        self.crop_size: int = crop_size
        self.mode: str = mode
        self.mel_spectrogram_param: dict[str, int] = mel_spectrogram_param
        self.precision: torch.dtype = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }.get(precision)
        self.sample_rate: int = 44100

        # self._init_transforms()

    def _init_transforms(self: Self, sample_rate: int) -> tuple[v2.Compose, v2.Compose]:
        transforms: list[nn.Module] = [
            T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate),
            T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.mel_spectrogram_param.get("N_FFT", 2048),
                # win_length=self.mel_spectrogram_param.get("WIN_LENGTH", 1024),
                hop_length=self.mel_spectrogram_param.get("HOP_LENGTH", 512),
                n_mels=self.mel_spectrogram_param.get("N_MELS", 256),
                normalized=False,
                power=2,
            ),
            T.AmplitudeToDB(top_db=80.0),
            # ZScoreNorm(),
            MinMaxNorm(min_value=0.0, max_value=1.0),
            v2.ToDtype(self.precision, scale=False),
        ]

        # self.y_transforms = v2.Compose(transforms)
        train_transforms = deepcopy(transforms)
        if self.mode == "train":
            train_transforms.insert(
                2, OneOf([T.TimeMasking(time_mask_param=100), T.FrequencyMasking(freq_mask_param=30)])
            )
        return v2.Compose(train_transforms), v2.Compose(transforms)

    # def waveform_transform(self: Self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    #     min_speed, max_speed = 0.5, 2.
    #     min_volume, max_volume = 0.2, 2.
    #     return OneOf([
    #         T.Speed(orig_freq=sample_rate, factor=(max_speed-min_speed)*torch.rand(1) + min_speed),
    #         T.Vol(gain=(max_volume-min_volume)*torch.rand(1) + min_volume)
    #     ])(waveform)

    def __len__(self: Self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_path)

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode != "train":
            return tensor
        noise = torch.randn_like(tensor)
        snr = torch.tensor((50, 50))
        return T.AddNoise()(tensor, noise, snr)

    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the item at the specified index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
        # print(self.data_path[index])
        with safe_open(self.data_path[index], framework="pt", device="cpu") as f:
            audio: torch.Tensor = f.get_tensor("audio")
            sample_rate: int = f.get_tensor("sample_rate").item()

        # print(audio.shape)
        x_transforms, y_transforms = self._init_transforms(sample_rate)

        num_frames: int = audio.shape[1]
        crop_frames: int = self.crop_size * sample_rate
        # print(f"num_frames: {num_frames}, crop_frames: {crop_frames}, sample_rate: {sample_rate}")
        if num_frames <= crop_frames:
            frames_to_add: int = crop_frames - audio.shape[1]
            audio = torch.cat([audio, torch.zeros((audio.shape[0], frames_to_add))], dim=1)
            return x_transforms(self.add_noise(audio)), y_transforms(audio)
            # return x_transforms(audio), y_transforms(audio)

        x_transformed: torch.Tensor
        y_transformed: torch.Tensor
        while True:
            frame_offset: int = random.randint(0, num_frames - crop_frames)
            cropped_audio = audio[:, frame_offset : frame_offset + crop_frames]
            x_transformed, y_transformed = x_transforms(self.add_noise(cropped_audio)), y_transforms(cropped_audio)
            # x_transformed, y_transformed = x_transforms(cropped_audio), y_transforms(cropped_audio)
            if not torch.isnan(x_transformed).sum() and not torch.isnan(x_transformed).sum():
                break
        return x_transformed, y_transformed
