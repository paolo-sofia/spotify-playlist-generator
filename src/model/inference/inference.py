import torch
import torchaudio
import torchaudio.transforms as T
from torchvision.transforms import v2

from src.db.schemas.song_embedding import SongEmbedding
from src.model.dataset.transforms import MinMaxNorm
from src.model.hyperparameters import Hyperparameters
from src.utils.config import config, load_model_hyperparameters


def define_transforms(config: Hyperparameters, sample_rate: int) -> v2.Compose:
    """Define the data transformations for audio input based on the provided configuration.

    Args:
        config: The hyperparameter configuration.

    Returns:
        v2.Compose: The Compose object containing data transformations.
    """
    precision: torch.dtype = {16: torch.float16, 32: torch.float32, 64: torch.float64}.get(
        config.PRECISION_INFERENCE, 32
    )

    return v2.Compose(
        [
            T.Resample(orig_freq=sample_rate, new_freq=config.SAMPLE_RATE),
            T.MelSpectrogram(
                sample_rate=config.SAMPLE_RATE,
                n_fft=config.N_FFT,
                win_length=config.WIN_LENGTH,
                hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS,
                normalized=False,
            ),
            MinMaxNorm(min_value=0.0, max_value=1.0),
            T.AmplitudeToDB(top_db=80.0),
            v2.ToDtype(precision, scale=False),
        ]
    )


def preprocess(tensor: torch.Tensor, transforms: v2.Compose) -> torch.Tensor:
    return transforms(tensor).to(device).unsqueeze(dim=0)


def load_model() -> torch.jit._script.RecursiveScriptModule:
    return torch.jit.load(config.get("model", {}).get("path"))


def wrap_prediction_to_song_embedding(tensor: torch.Tensor, song_id: str) -> SongEmbedding:
    return SongEmbedding(id=song_id, embedding=tensor.tolist())


def prepare_audio_waveform(audio: torch.Tensor, crop_frames: int) -> list[torch.Tensor]:
    num_frames: int = audio.shape[1]
    last_audio_slice_length: int = num_frames % crop_frames

    if last_audio_slice_length == 0:
        pass
    elif last_audio_slice_length > crop_frames / 2:
        frames_to_add: int = crop_frames - (num_frames % crop_frames)
        audio = torch.cat([audio, torch.zeros((audio.shape[0], frames_to_add))], dim=1)
    else:
        frames_to_remove: int = num_frames % crop_frames
        audio = audio[:, : num_frames - frames_to_remove]

    num_slices: int = audio.shape[1] // crop_frames
    return torch.chunk(audio, num_slices, dim=1)


def predict_audio(audio_chunks: list[torch.Tensor], transforms: v2.Compose) -> torch.Tensor:
    embeddings = torch.stack(
        [model.encoder(preprocess(audio_slice, transforms)) for audio_slice in audio_chunks]
    ).squeeze()

    if len(audio_chunks) == 1:
        return torch.concat([embeddings, torch.zeros_like(embeddings)])

    return torch.concat([embeddings.mean(dim=0), embeddings.std(dim=0)])


def get_song_embedding(track: dict[str, str]) -> SongEmbedding:
    song_id: str = track.get("song_id")
    audio_path: str = track.get("audio_path")
    audio, sample_rate = torchaudio.load(audio_path)

    num_frames: int = audio.shape[1]
    crop_frames: int = cfg.CROP_SIZE_SECONDS * sample_rate
    if num_frames <= crop_frames:
        frames_to_add: int = crop_frames * (num_frames // crop_frames + 1) - num_frames
        audio: torch.Tensor = torch.cat(tensors=[audio, torch.zeros((audio.shape[0], frames_to_add))], dim=1)
    audio_slices: list[torch.Tensor] = prepare_audio_waveform(audio=audio, crop_frames=crop_frames)

    transforms: v2.Compose = define_transforms(cfg, sample_rate)
    prediction: torch.Tensor = predict_audio(audio_slices, transforms)
    return wrap_prediction_to_song_embedding(prediction, song_id)


cfg: Hyperparameters = load_model_hyperparameters()
model: torch.jit._script.RecursiveScriptModule = load_model()
# device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "")
device: torch.device = torch.device("cpu")
