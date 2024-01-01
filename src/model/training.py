#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import random
import sys

import lightning
import numpy as np
import tomllib
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

sys.path.append(str(pathlib.Path.cwd()))

from src.model.autoencoder import Autoencoder, initialize_weights
from src.model.dataset.audio_dataset import AudioDataset
from src.model.dataset.utils import get_splits
from src.utils import Hyperparameters, dataclass_from_dict

torch.set_float32_matmul_precision("medium")


def seed_everything(seed: int) -> None:
    """Set the random seed for reproducibility in various libraries.

    Args:
        seed: The seed value to set.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


with (pathlib.Path.cwd() / "config" / "model.toml").open("rb") as f:
    cfg: Hyperparameters = dataclass_from_dict(Hyperparameters, tomllib.load(f))

seed_everything(cfg.SEED)

songs_path: list[pathlib.Path] = list(pathlib.Path(pathlib.Path.cwd()).parent.rglob("*.safetensors"))
train, valid, test = get_splits(
    data=songs_path, train_size=cfg.TRAIN_SIZE, valid_size=cfg.VALID_SIZE, test_size=cfg.TEST_SIZE, stratify_col=None
)


mel_spectrogram_params: dict[str, int] = {
    "N_FFT": cfg.N_FFT,
    "N_MELS": cfg.N_MELS,
    "WIN_LENGTH": cfg.WIN_LENGTH,
    "HOP_LENGTH": cfg.HOP_LENGTH,
}
train_dataloader = DataLoader(
    dataset=AudioDataset(
        data_path=train,
        mode="train",
        crop_size=cfg.CROP_SIZE_SECONDS,
        precision=cfg.PRECISION,
        mel_spectrogram_param=mel_spectrogram_params,
    ),
    batch_size=cfg.BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    persistent_workers=False,
)

valid_dataloader = DataLoader(
    dataset=AudioDataset(
        data_path=valid,
        mode="valid",
        crop_size=cfg.CROP_SIZE_SECONDS,
        precision=cfg.PRECISION,
        mel_spectrogram_param=mel_spectrogram_params,
    ),
    batch_size=cfg.BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    persistent_workers=False,
)

model = Autoencoder(hyperparam=cfg)
model.apply(initialize_weights)
model._log_hyperparams = False

savedir: pathlib.Path = pathlib.Path("/home/paolo/git/spotify-playlist-generator/logs/mlruns")
logger = MLFlowLogger(
    experiment_name="Autoencoder",
    save_dir=str(savedir),
    log_model=True,
    run_name="overfit batch" if cfg.OVERFIT_BATCHES else "Model resized no regularization",
)

early_stop_callback: EarlyStopping = EarlyStopping(
    monitor="valid_loss",
    min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
    patience=cfg.EARLY_STOPPING_PATIENCE,
    verbose=True,
    mode="min",
    check_finite=True,
)

model_checkpoint: ModelCheckpoint = ModelCheckpoint(
    dirpath=str(
        savedir / logger.experiment_id / logger.run_id / "artifacts" / "model" / "checkpoints" / "model_checkpoint"
    ),
    filename="model_checkpoint",
    monitor="valid_loss",
    verbose=True,
    save_last=None,
    save_top_k=1,
    save_weights_only=False,
    mode="min",
    auto_insert_metric_name=True,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
    enable_version_counter=True,
)

callbacks: list = [RichProgressBar()] if cfg.OVERFIT_BATCHES else [early_stop_callback, RichProgressBar()]
callbacks.append(model_checkpoint)

trainer: lightning.Trainer = lightning.Trainer(
    accelerator="gpu",
    num_nodes=1,
    precision=cfg.PRECISION,
    logger=logger,
    callbacks=callbacks,
    fast_dev_run=False,
    max_epochs=cfg.EPOCHS,
    min_epochs=1,
    overfit_batches=cfg.OVERFIT_BATCHES,
    log_every_n_steps=100,
    check_val_every_n_epoch=1,
    enable_checkpointing=True,
    enable_progress_bar=True,
    enable_model_summary=True,
    deterministic="warn",
    benchmark=True,
    inference_mode=True,
    profiler=None,  # AdvancedProfiler(),
    detect_anomaly=False,
    barebones=False,
    gradient_clip_val=cfg.GRADIENT_CLIP_VAL,
    gradient_clip_algorithm=cfg.GRADIENT_CLIP_TYPE,
    accumulate_grad_batches=cfg.GRADIENT_ACCUMULATION_BATCHES,
)

logger.log_hyperparams(cfg.__dict__)
# print(cfg.__dict__)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=None)
os.system("notify-send 'Training complete!'")
