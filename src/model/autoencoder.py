from __future__ import annotations

from typing import Self

import lightning
import torch
from optimizers.yogi import Yogi
from torch import nn, optim

from src.utils import Hyperparameters


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
        input_shape: int,
        num_input_channels: int,
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
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, base_channel_size, kernel_size=3, padding=1, stride=2),  # 256 => 128
            # nn.LayerNorm([base_channel_size, input_shape // 2**1, input_shape // 2**1]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.Conv2d(base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2),  # 128 => 64
            # nn.LayerNorm([base_channel_size * 2, input_shape // 2**2, input_shape // 2**2]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.Conv2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2
            ),  # 64 => 32,32,32
            # nn.LayerNorm([base_channel_size * 2, input_shape // 2**3, input_shape // 2**3]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.Conv2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2
            ),  # 32 => 32,16,16
            # nn.LayerNorm([base_channel_size * 2, input_shape // 2**4, input_shape // 2**4]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.Flatten(),
            nn.Linear(512 * base_channel_size, latent_dim),
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
        input_shape: int,
        num_input_channels: int,
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
        self.linear = nn.Sequential(nn.Linear(latent_dim, 512 * base_channel_size), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2, output_padding=1
            ),  # 4x4 => 8x8
            # nn.LayerNorm([base_channel_size * 2, input_shape // 2**3, input_shape // 2**3]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.ConvTranspose2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2, output_padding=1
            ),  # 8x8 => 16x16
            # nn.LayerNorm([base_channel_size * 2, input_shape // 2**2, input_shape // 2**2]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.ConvTranspose2d(
                2 * base_channel_size, base_channel_size, kernel_size=3, padding=1, stride=2, output_padding=1
            ),  # 16x16 => 32x32
            # nn.LayerNorm([base_channel_size, input_shape // 2, input_shape // 2]),
            # torchvision.ops.drop_block.DropBlock2d(p=0.3, block_size=5),
            act_fn(),
            nn.ConvTranspose2d(
                base_channel_size, num_input_channels, kernel_size=3, padding=1, stride=2, output_padding=1
            ),  # 16x16 => 32x32
        )

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the network and returns the output.

        Args:
            inputs: The input tensor to be passed through the network.

        Returns:
            The output tensor produced by the network.
        """
        x = self.linear(inputs)
        x = x.reshape(x.shape[0], -1, 16, 16)
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
        loss: nn.Module = nn.L1Loss,
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
            input_shape=self.hyperparam.INPUT_SIZE,
            num_input_channels=self.hyperparam.NUM_INPUT_CHANNELS,
            base_channel_size=self.hyperparam.BASE_CHANNEL_SIZE,
            latent_dim=self.hyperparam.LATENT_DIM,
            act_fn=nn.Mish,
        )
        self.decoder = Decoder(
            input_shape=self.hyperparam.INPUT_SIZE,
            num_input_channels=self.hyperparam.NUM_INPUT_CHANNELS,
            base_channel_size=self.hyperparam.BASE_CHANNEL_SIZE,
            latent_dim=self.hyperparam.LATENT_DIM,
            act_fn=nn.Mish,
        )
        self.loss = loss(reduction="mean")  # nn.MSELoss(reduction="mean")

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the network and returns the output.

        Args:
            inputs: The input tensor to be passed through the network.

        Returns:
            The output tensor produced by the network.
        """
        z = self.encoder(inputs)
        return self.decoder(z)

    def configure_optimizers(self: Self) -> dict[str, optim.Optimizer | optim.lr_scheduler.LRScheduler | str]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            A dictionary containing the optimizer, learning rate scheduler, and monitor metric.
        """
        optimizer: optim.Optimizer = Yogi(
            params=self.parameters(), lr=self.hyperparam.LEARNING_RATE, weight_decay=self.hyperparam.LEARNING_RATE_DECAY
        )

        scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=self.hyperparam.T_0, T_mult=self.hyperparam.T_mult
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}

    def _log_metrics(self: Self, loss: float, step: str = "train") -> None:
        """Logs the specified loss metric during training or validation.

        Args:
            loss: The value of the loss metric to be logged.
            step: The step or phase of the training or validation process. Defaults to "train".
        """
        self.log(f"{step}_loss", loss, prog_bar=True, logger=False, on_epoch=True, on_step=False)
        self.logger.log_metrics({f"{step}_loss": loss.detach().item()}, step=self.current_epoch + 1)

    def training_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Performs a training step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="train")

    def validation_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Performs a validation step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="valid")

    def test_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
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
        x_hat = self.forward(x)
        loss = self.loss(x_hat, y)
        self._log_metrics(loss=loss, step=step)
        return loss


def initialize_weights(layer: nn.Module) -> None:
    """Initializes the weights of the network.

    Args:
        layer: The layer of the network.
    """
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_uniform_(layer.weight.data, a=0.0003, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight.data, 1)
        nn.init.constant_(layer.bias.data, 0)
