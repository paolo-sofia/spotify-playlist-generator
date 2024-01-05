import dataclasses
from typing import Self

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
    SAMPLE_RATE: int
    SEED: int
    GRADIENT_CLIP_VAL: float
    GRADIENT_CLIP_TYPE: str
    PRECISION: int
    GRADIENT_ACCUMULATION_BATCHES: int
    LOSS: nn.Module = nn.L1Loss
    CROP_FRAMES: int = 0
    N_FFT: int = 512
    WIN_LENGTH: int = 512
    HOP_LENGTH: int = 256
    N_MELS: int = 256

    def __post_init__(self: Self) -> None:
        """Calculate the number of frames to crop based on the specified crop size in seconds and sample rate.

        Args:
            self: The instance of the class.

        Returns:
            None
        """
        self.CROP_FRAMES = self.CROP_SIZE_SECONDS * self.SAMPLE_RATE
