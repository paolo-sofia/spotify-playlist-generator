import dataclasses

from torch import nn


@dataclasses.dataclass
class Hyperparameters:
    """Data class for storing hyperparameters used in training.

    Attributes:
        BATCH_SIZE (int): The batch size for training.
        EPOCHS (int): The number of training epochs.
        LEARNING_RATE (float): The learning rate for the optimizer.
        WEIGHT_DECAY (int): The weight decay of the optimizer, if supported.
        TRAIN_SIZE (float): The proportion of the data used for training.
        BASE_CHANNEL_SIZE (int): The base number of channels for the model.
        LATENT_DIM (int): The dimension of the latent space.
        NUM_INPUT_CHANNELS (int): The number of input channels.
        T_0 (int): The T_0 parameter for the learning rate scheduler.
        T_mult (int): The T_mult parameter for the learning rate scheduler.
        EARLY_STOPPING_PATIENCE (int): The patience for early stopping.
        EARLY_STOPPING_MIN_DELTA (float): The minimum delta for early stopping.
        OVERFIT_BATCHES (int): The number of batches to over fit on.
        LOSS (nn.Module): The loss function used for training.
        CROP_SIZE_SECONDS (int): The size of the cropped audio in seconds.
        SEED (int): The random seed for reproducibility.
        PRECISION (int): The floating point precision used for training.
        GRADIENT_ACCUMULATION_BATCHES (int): The number of batches to run before updating the weights
    """

    SEED: int
    TRAIN_SIZE: float
    VALID_SIZE: float
    TEST_SIZE: float
    CROP_SIZE_SECONDS: int
    SAMPLE_RATE: int
    BATCH_SIZE: int
    NUM_INPUT_CHANNELS: int
    BASE_CHANNEL_SIZE: int
    LATENT_DIM: int
    EPOCHS: int
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    T_0: int
    T_mult: int
    EARLY_STOPPING_PATIENCE: int
    EARLY_STOPPING_MIN_DELTA: float
    OVERFIT_BATCHES: int
    GRADIENT_CLIP_VAL: float
    GRADIENT_CLIP_TYPE: str = "norm"
    PRECISION: int = 16
    PRECISION_INFERENCE: int = 32
    GRADIENT_ACCUMULATION_BATCHES: int = 1
    LOSS: nn.Module = nn.L1Loss
    N_FFT: int = 512
    WIN_LENGTH: int = 512
    HOP_LENGTH: int = 256
    N_MELS: int = 256
