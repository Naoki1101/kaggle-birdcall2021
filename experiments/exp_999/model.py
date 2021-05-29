from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
import kornia.augmentation as aug

from src import layer


class CustomModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        model_name: str,
        args_spec: Dict,
        in_channels: int = 1,
        pooling_name: str = "GeM",
        args_pooling: Optional[Dict] = None,
    ):
        super(CustomModel, self).__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=args_spec.n_fft,
            hop_length=args_spec.hop_length,
            win_length=args_spec.n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=args_spec.sampling_rate,
            n_fft=args_spec.n_fft,
            n_mels=args_spec.n_mels,
            fmin=args_spec.fmin,
            fmax=args_spec.fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.transform = aug.GaussianBlur((3, 3), (0.1, 3.0), p=1.0)

        self.bn0 = nn.BatchNorm2d(args_spec.n_mels)
        # self.bn0 = nn.LayerNorm([313, 128])

        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=in_channels
        )

        final_in_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(layer, pooling_name)(**args_pooling)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(final_in_features, n_classes)

    def forward(self, x, is_train):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # x = self.transform(x)

        # max_ = torch.amax(x, dim=(2, 3), keepdim=True)
        # min_ = torch.amin(x, dim=(2, 3), keepdim=True)
        # x = (x - min_) / (max_ - min_ + 1e-5)

        if is_train:
            x = self.spec_augmenter(x)

        x = x.contiguous().transpose(2, 3)

        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(len(x), -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
