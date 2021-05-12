import sys
from typing import Dict

import numpy as np
import pandas as pd

import librosa
import audiomentations as audio
from torch.utils.data import Dataset

sys.path.append("./src")
import const


def get_transforms(params: Dict):
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


class CustomTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["filename"].values
        self.labels = df["target"].values.reshape(-1, 1)
        self.transforms = get_transforms(cfg.transforms)
        self.primary_label = df["primary_label"].values

        self.samples = const.TARGET_SAMPLE_RATE * 5

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
            "pink": pink_noise,
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        noise_r = np.random.rand()

        if noise_r <= 0.5:
            filename = self.filenames[idx]
            filename = filename.replace(".ogg", ".wav")
            primary_label = self.primary_label[idx]
            path_name = str(const.TRAIN_RESAMPLED_AUDIO_DIR / primary_label / filename)

            y, sr = librosa.load(path_name, sr=const.TARGET_SAMPLE_RATE)

            len_y = len(y)
            if len_y < self.samples:
                padding = self.samples - len_y
                offset = padding // 2
                y = np.pad(
                    y, (offset, self.samples - len_y - offset), "constant"
                ).astype(np.float32)
            elif len_y > self.samples:
                y = y[: self.samples].astype(np.float32)
            else:
                y = y.astype(np.float32)
            label = self.labels[idx, :]
        else:
            y = np.zeros(self.samples).astype(np.float32)
            label = np.array([0.0], dtype=np.float32)

        rand = np.random.rand()
        m = np.random.rand() * 50
        for noise_name in self.cfg.noise:
            noise_threshhold = getattr(self.cfg.noise, noise_name)
            noise_wave = self.noise_dict[noise_name]
            if rand >= noise_threshhold[0] and rand < noise_threshhold[1]:
                start = np.random.randint(len(noise_wave) - self.samples)
                y += noise_wave[start : start + self.samples].astype(np.float32) * m

        if self.transforms:
            y = self.transforms(samples=y, sample_rate=const.TARGET_SAMPLE_RATE)

        return y, label


class CustomValidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["file_name"].values
        self.seconds = df["seconds"].values
        self.labels = df["target"].values.reshape(-1, 1)
        self.transforms = get_transforms(cfg.transforms)
        self.audio_dict = {}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        seconds = self.seconds[idx]
        path_name = str(const.TRAIN_SOUNDSCAPES_DIR / filename)

        if filename not in self.audio_dict:
            y, sr = librosa.load(path_name, sr=const.TARGET_SAMPLE_RATE)
            self.audio_dict[filename] = y
        else:
            y = self.audio_dict[filename]

        start_index = const.TARGET_SAMPLE_RATE * (seconds - 5)
        end_index = const.TARGET_SAMPLE_RATE * seconds
        y = y[start_index:end_index].astype(np.float32)

        if self.transforms:
            y = self.transforms(samples=y, sample_rate=const.TARGET_SAMPLE_RATE)

        label = self.labels[idx]

        return y, label
