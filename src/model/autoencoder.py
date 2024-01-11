from __future__ import annotations

from typing import Self

import lightning
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from .hyperparameters import Hyperparameters
from .optimizers.adabelief import AdaBelief


class CustomLoss(nn.Module):
    def __init__(self: Self, penalty_weight: float = 1.0, reduction: str = "mean"):
        super(CustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.reduction = reduction

    def forward(self, predictions, targets) -> torch.Tensor:
        l1_loss: torch.Tensor = F.l1_loss(predictions, targets, reduction=self.reduction)
        mse_loss: torch.Tensor = F.mse_loss(predictions, targets, reduction=self.reduction)
        return l1_loss + self.penalty_weight * mse_loss


class PeriodicReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, torch.cos(x) * x, torch.sin(x))


class Encoder(nn.Module):
    """Encoder.

    Args:
       num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
       base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a
        duplicate of it.
       latent_dim : Dimensionality of latent representation z
       act_fn : Activation function used throughout the encoder network
    """

    def __init__(
        self: Self,
        num_input_channels: int,
        input_shape: tuple[int, ...],
        base_channel_size: int,
        latent_dim: int,
        act_fn: nn.Module,
    ) -> None:
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a
            duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.conv = nn.Conv2d(num_input_channels, base_channel_size, kernel_size=3, padding=1, stride=2)
        self.act_fn = act_fn
        self.base_channel_size = base_channel_size
        self.num_input_channels = num_input_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 3),
                padding=(1, 3),
                stride=(2, 2),
            ),
            # nn.LayerNorm([self.base_channel_size * 2, self.input_shape[1] // 2**1, self.input_shape[2] // 2**1]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.Conv2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 5),
                padding=(1, 1),
                stride=(2, 3),
            ),
            # nn.LayerNorm([self.base_channel_size * 2, self.input_shape[1] // 2**2, self.input_shape[2] // (2**2 + 2)]),
            act_fn(),
            nn.Conv2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 5),
                padding=(1, 1),
                stride=(2, 3),
            ),
            # nn.LayerNorm([self.base_channel_size * 2, self.input_shape[1] // 2**3, self.input_shape[2] // (2**4 + 2)]),
            act_fn(),
            nn.Conv2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 5),
                padding=(1, 1),
                stride=(2, 4),
            ),
            act_fn(),
            nn.Flatten(),
            nn.Linear(in_features=576 * self.base_channel_size, out_features=self.latent_dim),
            act_fn(),
        )

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the network and returns the output.

        Args:
            inputs: The input tensor to be passed through the network.

        Returns:
            The output tensor produced by the network.
        """
        return self.net(inputs)


class Decoder(nn.Module):
    """Decoder.

    Args:
       num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
       base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
       latent_dim : Dimensionality of latent representation z
       act_fn : Activation function used throughout the decoder network
    """

    def __init__(
        self: Self,
        num_input_channels: int,
        input_shape: tuple[int, ...],
        base_channel_size: int,
        latent_dim: int,
        act_fn: nn.Module,
    ) -> None:
        """Decoder.

        Args:
           input_shape: the shape of the input tensor
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.act_fn = act_fn
        self.base_channel_size = base_channel_size
        self.num_input_channels = num_input_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.linear = nn.Sequential(
            nn.Linear(self.latent_dim, self.base_channel_size * 576),
            act_fn(),
            nn.Unflatten(dim=1, unflattened_size=(self.base_channel_size * 2, 16, 18)),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 5),
                padding=(1, 1),
                stride=(2, 4),
                output_padding=(1, 1),
            ),
            # nn.LayerNorm([self.base_channel_size * 2, self.input_shape[1] // 2**3, self.input_shape[2] // (2**4 + 2)]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.ConvTranspose2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 5),
                padding=(1, 1),
                stride=(2, 3),
                output_padding=(1, 0),
            ),
            # nn.LayerNorm([self.base_channel_size * 2, self.input_shape[1] // 2**2, self.input_shape[2] // (2**2 + 2)]),
            act_fn(),
            nn.ConvTranspose2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.base_channel_size * 2,
                kernel_size=(3, 5),
                padding=(1, 1),
                stride=(2, 3),
                output_padding=(1, 0),
            ),
            # nn.LayerNorm([self.base_channel_size * 2, self.input_shape[1] // 2**1, self.input_shape[2] // 2**1]),
            act_fn(),
            nn.ConvTranspose2d(
                in_channels=self.base_channel_size * 2,
                out_channels=self.num_input_channels,
                kernel_size=(3, 3),
                padding=(1, 3),
                stride=(2, 2),
                output_padding=(1, 1),
            ),
            # nn.Sigmoid(),
        )

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the network and returns the output.

        Args:
            inputs: The input tensor to be passed through the network.

        Returns:
            The output tensor produced by the network.
        """
        x = self.linear(inputs)
        return self.net(x)


class Autoencoder(lightning.LightningModule):
    """Initializes the Autoencoder model with the specified hyperparameters.

    Args:
        hyperparam: The hyperparameters of the autoencoder.
        loss: The loss function used for training the autoencoder.
    """

    def __init__(
        self: Self,
        hyperparam: Hyperparameters,
        loss: nn.Module = CustomLoss,
    ) -> None:
        """Initializes the Autoencoder model.

        Args:
            hyperparam: The hyperparameters of the autoencoder.
            loss: The loss function used for training the autoencoder.

        Returns:
            None
        """
        super().__init__()
        self.hyperparam = hyperparam
        self.encoder = Encoder(
            input_shape=(self.hyperparam.NUM_INPUT_CHANNELS, self.hyperparam.N_MELS, 1188),
            num_input_channels=self.hyperparam.NUM_INPUT_CHANNELS,
            base_channel_size=self.hyperparam.BASE_CHANNEL_SIZE,
            latent_dim=self.hyperparam.LATENT_DIM,
            act_fn=nn.Mish,
        )
        self.decoder = Decoder(
            num_input_channels=self.hyperparam.NUM_INPUT_CHANNELS,
            input_shape=(self.hyperparam.NUM_INPUT_CHANNELS, self.hyperparam.N_MELS, 1188),
            base_channel_size=self.hyperparam.BASE_CHANNEL_SIZE,
            latent_dim=self.hyperparam.LATENT_DIM,
            act_fn=nn.Mish,
        )
        self.loss = loss()
        self.metrics = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the network and returns the output.

        Args:
            inputs: The input tensor to be passed through the network.

        Returns:
            The output tensor produced by the network.
        """
        z = self.encoder(inputs)
        return self.decoder(z)

    def configure_optimizers(self: Self) -> optim.Optimizer:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            A dictionary containing the optimizer, learning rate scheduler, and monitor metric.
        """
        return AdaBelief(
            params=self.parameters(),
            lr=self.hyperparam.LEARNING_RATE,
            weight_decay=self.hyperparam.LEARNING_RATE_DECAY,
            weight_decouple=True,
            rectify=False,
            fixed_decay=False,
            amsgrad=False,
        )

    def _log_metrics(
        self: Self,
        loss: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        pred: torch.Tensor,
        y_true: torch.Tensor,
        step: str = "train",
    ) -> None:
        """Logs the specified loss metric during training or validation.

        Args:
            loss: The value of the loss metric to be logged.
            step: The step or phase of the training or validation process. Defaults to "train".
        """
        self.log(f"{step}_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.logger.log_metrics(
            {f"{step}_{k}": metric.detach().item() for k, metric in metrics.items()}, step=self.global_step
        )
        self.logger.log_metrics(
            {"pred_mean": pred, "real_mean": y_true, "mean_diff": y_true - pred}, step=self.global_step
        )

    def training_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Performs a training step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="train")

    def validation_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Performs a validation step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="valid")

    def test_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Performs a validation step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="test")

    def _forward(self: Self, batch: tuple[torch.Tensor, ...], step: str) -> torch.Tensor:
        """Performs a forward pass on a batch of data and calculates the loss.

        Args:
            batch: The input batch of data.
            step: The step or phase of the forward pass.

        Returns:
            The calculated loss value.
        """
        x, y = batch
        x_hat: torch.Tensor = self.forward(x)
        loss: torch.Tensor = self.loss(x_hat, y)
        metrics: dict[str, torch.Tensor] = {k: metric.to(self.device)(x_hat, y) for k, metric in self.metrics.items()}

        self._log_metrics(
            loss=loss, step=step, metrics=metrics, pred=x_hat.mean().detach().item(), y_true=y.mean().detach().item()
        )
        return loss


def initialize_weights(layer: nn.Module) -> None:
    """Initializes the weights of the network.

    Args:
        layer: The layer of the network.
    """
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        (nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain("relu")),)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight.data, 1)
        nn.init.constant_(layer.bias.data, 0)
