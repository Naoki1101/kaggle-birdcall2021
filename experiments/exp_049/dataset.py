import sys
from typing import Dict

import numpy as np
import pandas as pd

import librosa
import audiomentations as audio
from torch.utils.data import Dataset

sys.path.append("./src")
import const
from utils import DataHandler

dh = DataHandler()


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
    def __init__(self, df: pd.DataFrame, target_df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["filename"].values
        self.labels = target_df.values[:, :-1].astype(float)
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

        self.noise_dict = {
            "nocall": example_noise,
            "water": water_noise,
            "bus": bus_noise,
            "walk": walk_noise,
            "rain": rain_noise,
            "motorcycle": motorcycle_noise,
        }

        self.train_noise_dict = dh.load(const.PROCESSED_DATA_DIR / "train_noise.json")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filename_ = filename.replace(".ogg", ".wav")
        primary_label = self.primary_label[idx]
        path_name = str(const.TRAIN_RESAMPLED_AUDIO_DIR / primary_label / filename_)

        y, sr = librosa.load(path_name, sr=const.TARGET_SAMPLE_RATE)

        len_y = len(y)
        if len_y < self.samples:
            padding = self.samples - len_y
            offset = padding // 2
            y = np.pad(y, (offset, self.samples - len_y - offset), "constant").astype(
                np.float32
            )
        elif len_y >= self.samples:
            if filename in self.train_noise_dict:
                noise_chunk_list = self.train_noise_dict[filename]
            else:
                noise_chunk_list = []

            for _ in range(5):
                start = np.random.randint(len_y - self.samples)
                noise = self._check_noise(start, noise_chunk_list)
                if not noise:
                    break

            y = y[start : start + self.samples].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.cfg.noise:
            rand = np.random.rand()
            m = (np.random.rand() + 0.2) * 10

            for noise_name in self.cfg.noise:
                noise_threshhold = getattr(self.cfg.noise, noise_name)
                noise_wave = self.noise_dict[noise_name]
                if rand >= noise_threshhold[0] and rand < noise_threshhold[1]:
                    start = np.random.randint(len(noise_wave) - self.samples)
                    y += noise_wave[start : start + self.samples].astype(np.float32) * m

        if self.transforms:
            y = self.transforms(samples=y, sample_rate=const.TARGET_SAMPLE_RATE)

        label = self.labels[idx, :]

        return y, label

    def _check_noise(self, start_time, noise_list):
        result = False

        round_s = start_time // const.TARGET_SAMPLE_RATE
        if round_s % 5 == 0:
            round_e = round_s
        else:
            round_e = round_s - (round_s % 5) + 5

        if round_e in noise_list:
            result = True

        return result


class CustomValidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["file_name"].values
        self.seconds = df["seconds"].values
        self.labels = target_df.values[:, :-1].astype(float)
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

        label = self.labels[idx, :]
        return y, label
