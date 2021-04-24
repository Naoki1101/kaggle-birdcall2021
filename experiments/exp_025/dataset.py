import sys
from typing import Dict

import numpy as np
import pandas as pd
import cv2
import librosa
import albumentations as album
import torch
from torch.utils.data import Dataset

sys.path.append("./src")
import const
from utils import DataHandler

dh = DataHandler()


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


def get_transforms(params: Dict):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        else:
            return eval(transform.name)

    transforms = None
    if params is not None:
        transforms = [
            get_object(transform)(**transform.params)
            for name, transform in params.items()
        ]
        transforms = album.Compose(transforms)

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
        self.transforms = get_transforms(cfg.transforms)
        self.primary_label = df["primary_label"].values

        distance_array = np.load(
            const.PROCESSED_DATA_DIR / "distance_array_from_min_position.npy"
        )
        self.distances = np.log1p(distance_array)

        example_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "train_soundscape_nocall.wav",
            sr=conf.sampling_rate,
        )
        water_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_water_noise.wav",
            sr=conf.sampling_rate,
        )
        bus_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_bus_noise.wav",
            sr=conf.sampling_rate,
        )
        walk_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_walk_noise.wav",
            sr=conf.sampling_rate,
        )
        rain_noise, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_rain_noise.wav",
            sr=conf.sampling_rate,
        )
        motorcycle, _ = librosa.load(
            const.NOISE_AUDIO_DIR / "freesound_motorcycle_noise.wav",
            sr=conf.sampling_rate,
        )

        self.noise_dict = {
            "nocall": example_noise,
            "water": water_noise,
            "bus": bus_noise,
            "walk": walk_noise,
            "rain": rain_noise,
            "motorcycle": motorcycle,
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
            m = np.random.randint(1, 10)

            for noise_name in self.cfg.noise:
                noise_threshhold = getattr(self.cfg.noise, noise_name)
                noise_wave = self.noise_dict[noise_name]
                if rand >= noise_threshhold[0] and rand < noise_threshhold[1]:
                    start = np.random.randint(len(noise_wave) - conf.samples)
                    y += noise_wave[start : start + conf.samples].astype(np.float32) * m

        melspec = librosa.feature.melspectrogram(
            y,
            sr=conf.sampling_rate,
            n_mels=conf.n_mels,
            fmin=conf.fmin,
            fmax=conf.fmax,
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = cv2.resize(image, (self.cfg.img_size.height, self.cfg.img_size.width))
        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        label = self.labels[idx, :]
        distance = self.distances[idx, :]

        feats = {
            "image": torch.FloatTensor(image),
            "distance": torch.FloatTensor(distance),
        }

        return feats, label


class CustomValidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_df: pd.DataFrame, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df["file_name"].values
        self.seconds = df["seconds"].values
        self.labels = target_df.values[:, :-1].astype(float)
        self.transforms = get_transforms(cfg.transforms)
        self.audio_dict = {}

        distance_array = np.load(
            const.PROCESSED_DATA_DIR / "valid_distance_array_from_min_position.npy"
        )
        distance_df = pd.DataFrame(
            distance_array,
            index=const.POS_DICT.keys(),
            columns=list(const.BIRD_CODE.keys())[:-1],
        )

        for col in distance_df.columns:
            df[f"distance_{col}"] = df["site"].map(distance_df[col])

        self.distances = np.log1p(df[df.columns[-397:]].values)

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

        melspec = librosa.feature.melspectrogram(
            y,
            sr=conf.sampling_rate,
            n_mels=conf.n_mels,
            fmin=conf.fmin,
            fmax=conf.fmax,
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = cv2.resize(image, (self.cfg.img_size.height, self.cfg.img_size.width))
        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        label = self.labels[idx, :]
        distance = self.distances[idx, :]

        feats = {
            "image": torch.FloatTensor(image),
            "distance": torch.FloatTensor(distance),
        }

        return feats, label
