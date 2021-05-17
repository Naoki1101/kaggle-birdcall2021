import sys
from typing import Dict

import numpy as np
import pandas as pd

import cv2
import librosa
import audiomentations as audio
from torch.utils.data import Dataset

sys.path.append("./src")
import const


class conf:
    duration = 5
    sampling_rate = 32_000
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = sampling_rate // 2
    power = 2.0
    samples = sampling_rate * duration


def get_audio_transforms(params: Dict):
    def get_object(transform):
        if hasattr(audio, transform.name):
            return getattr(audio, transform.name)
        else:
            return eval(transform.name)

    transforms = None
    if params is not None:
        transforms = [
            get_object(transform)(**transform.params)
            for name, transform in params.items()
        ]
        transforms = audio.Compose(transforms)

    return transforms


def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class CustomTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["filename"].values
        self.labels = target_df.values[:, :-1].astype(float)
        self.audio_transforms = get_audio_transforms(cfg.transforms)
        self.primary_label = df["primary_label"].values

        example_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "train_soundscape_nocall.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        water_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_water_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        bus_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_bus_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        walk_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_walk_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        rain_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_rain_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        motorcycle_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_motorcycle_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        speaking_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_speaking_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        random_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_random_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )
        pink_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "pink_noise.wav",
            sr=const.TARGET_SAMPLE_RATE,
        )

        self.noise_dict = {
            "nocall": example_noise,
            "water": water_noise,
            "bus": bus_noise,
            "walk": walk_noise,
            "rain": rain_noise,
            "motorcycle": motorcycle_noise,
            "speaking": speaking_noise,
            "random": random_noise,
            "pink": pink_noise,
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filename = filename.replace(".ogg", ".wav")
        primary_label = self.primary_label[idx]
        path_name = str(const.TRAIN_RESAMPLED_AUDIO_DIR / primary_label / filename)

        y, sr = librosa.load(path_name, sr=conf.sampling_rate)

        len_y = len(y)
        if len_y < conf.samples:
            padding = conf.samples - len_y
            offset = padding // 2
            y = np.pad(y, (offset, conf.samples - len_y - offset), "constant").astype(
                np.float32
            )
        elif len_y > conf.samples:
            y = y[: conf.samples].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.cfg.shift:
            rand = np.random.rand()
            if rand >= 0.5:
                shift = np.random.randint(conf.samples)
                y = np.roll(y, shift)

        if self.cfg.noise:
            rand = np.random.rand()
            m = np.random.rand() * 10

            for noise_name in self.cfg.noise:
                noise_threshhold = getattr(self.cfg.noise, noise_name)
                noise_wave = self.noise_dict[noise_name]
                if rand >= noise_threshhold[0] and rand < noise_threshhold[1]:
                    start = np.random.randint(len(noise_wave) - conf.samples)
                    y += noise_wave[start : start + conf.samples].astype(np.float32) * m

        if self.audio_transforms:
            y = self.audio_transforms(samples=y, sample_rate=const.TARGET_SAMPLE_RATE)

        melspec = librosa.feature.melspectrogram(
            y,
            sr=conf.sampling_rate,
            n_mels=conf.n_mels,
            fmin=conf.fmin,
            fmax=conf.fmax,
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)

        # if self.img_transforms:
        #     image = self.img_transforms(image=image)["image"]

        image = cv2.resize(image, (self.cfg.img_size.height, self.cfg.img_size.width))
        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        label = self.labels[idx, :]
        return image, label


class CustomValidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["file_name"].values
        self.seconds = df["seconds"].values
        self.labels = target_df.values[:, :-1].astype(float)
        self.audio_transforms = get_audio_transforms(cfg.transforms)
        self.audio_dict = {}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        seconds = self.seconds[idx]
        path_name = str(const.TRAIN_SOUNDSCAPES_DIR / filename)

        if filename not in self.audio_dict:
            y, sr = librosa.load(path_name, sr=conf.sampling_rate)
            self.audio_dict[filename] = y
        else:
            y = self.audio_dict[filename]

        start_index = conf.sampling_rate * (seconds - 5)
        end_index = conf.sampling_rate * seconds
        y = y[start_index:end_index].astype(np.float32)

        if self.audio_transforms:
            y = self.audio_transforms(samples=y, sample_rate=const.TARGET_SAMPLE_RATE)

        melspec = librosa.feature.melspectrogram(
            y,
            sr=conf.sampling_rate,
            n_mels=conf.n_mels,
            fmin=conf.fmin,
            fmax=conf.fmax,
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)

        # if self.img_transforms:
        #     image = self.img_transforms(image=image)["image"]

        image = cv2.resize(image, (self.cfg.img_size.height, self.cfg.img_size.width))
        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        label = self.labels[idx, :]
        return image, label
