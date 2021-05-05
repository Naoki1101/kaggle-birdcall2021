import warnings
from typing import Dict, Optional

import numpy as np
import librosa
import timm
import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram
from torchlibrosa.augmentation import SpecAugmentation

from src import layer


# source: https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/stft.py
class CustomLogmelFilterBank(nn.Module):
    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        n_mels=64,
        fmin=0.0,
        fmax=None,
        is_log=True,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
        freeze_parameters=True,
    ):
        r"""Calculate logmel spectrogram using pytorch. The mel filter bank is
        the pytorch implementation of as librosa.filters.mel
        """
        super(CustomLogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax is None:
            fmax = sr // 2

        self.melW = self.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.
        Args:
            input: (*, n_fft), spectrogram

        Returns:
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError(
                    "top_db must be non-negative"
                )
            log_spec = torch.clamp(
                log_spec, min=log_spec.max().item() - self.top_db, max=np.inf
            )

        return log_spec

    # source: https://github.com/yusuke10sato/bird_call/blob/master/src/train.py
    def mel(self, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1):
        """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

        Parameters
        ----------
        sr        : number > 0 [scalar]
            sampling rate of the incoming signal

        n_fft     : int > 0 [scalar]
            number of FFT components

        n_mels    : int > 0 [scalar]
            number of Mel bands to generate

        fmin      : float >= 0 [scalar]
            lowest frequency (in Hz)

        fmax      : float >= 0 [scalar]
            highest frequency (in Hz).
            If `None`, use `fmax = sr / 2.0`

        htk       : bool [scalar]
            use HTK formula instead of Slaney

        norm : {None, 1, np.inf} [scalar]
            if 1, divide the triangular mel weights by the width of the mel band
            (area normalization).  Otherwise, leave all the triangles aiming for
            a peak value of 1.0

        Returns
        -------
        M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
            Mel transform matrix

        Notes
        -----
        This function caches at level 10.

        Examples
        --------
        >>> melfb = librosa.filters.mel(22050, 2048)
        >>> melfb
        array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
            ...,
            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
            [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


        Clip the maximum frequency to 8KHz

        >>> librosa.filters.mel(22050, 2048, fmax=8000)
        array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
            ...,
            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
            [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


        >>> import matplotlib.pyplot as plt
        >>> plt.figure()
        >>> librosa.display.specshow(melfb, x_axis='linear')
        >>> plt.ylabel('Mel filter')
        >>> plt.title('Mel filter bank')
        >>> plt.colorbar()
        >>> plt.tight_layout()
        """

        if fmax is None:
            fmax = float(sr) / 2

        # if norm is not None and norm != 1 and norm != np.inf:
        #     raise ParameterError("Unsupported norm: {}".format(repr(norm)))

        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)))

        # Center freqs of each FFT bin
        fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = np.array(
            [
                200,
                200.65839880255413,
                201.37750848432012,
                201.51126380583162,
                201.67386956899037,
                201.87582543688856,
                202.13345527203808,
                202.47362290912503,
                202.94395795945513,
                203.63802229847101,
                204.7696777583669,
                206.9686321525935,
                212.88885671471746,
                217.43048869703566,
                217.67573891228102,
                217.99718059148606,
                218.43718046012455,
                219.07709875345563,
                220.09657180388132,
                221.99293137020203,
                226.94388589196586,
                232.62734674169212,
                235.10231672457329,
                236.44419592188694,
                239.27104445621455,
                247.84189145158842,
                258.1332636574189,
                266.45592538764413,
                287.9890352180489,
                310.53399991020956,
                329.7280481648527,
                365.2636543430171,
                412.1971069913186,
                474.9992071267056,
                543.8012938633804,
                613.9981630065175,
                684.4309008264818,
                746.926859193976,
                824.9481206944652,
                934.1647408605522,
                1059.406578152521,
                1191.4714986828917,
                1315.3661976656404,
                1417.0854269491579,
                1511.4688454929535,
                1612.80056993873,
                1721.6428871330952,
                1838.288761545141,
                1955.3394189452179,
                2064.968083010285,
                2166.8084495171515,
                2260.0916983488605,
                2338.3314875237083,
                2401.906399148228,
                2463.7128966413247,
                2524.8571234176743,
                2580.0580920624607,
                2628.9702036006215,
                2682.628540587129,
                2735.574322816629,
                2782.2949980039793,
                2823.2077006567883,
                2863.515614399661,
                2908.3053773334987,
                2947.9254283746377,
                2987.2898160362897,
                3026.660057289936,
                3066.3058926403455,
                3110.9589651232495,
                3150.8433299423505,
                3190.7647125250114,
                3235.4147868690166,
                3275.3314141881165,
                3315.610520650735,
                3360.6813727382987,
                3400.707128832307,
                3440.631288594453,
                3485.2582072693517,
                3525.048226029132,
                3565.061232929195,
                3610.3530117659466,
                3656.158234266066,
                3702.626404589744,
                3749.713173618707,
                3797.136734159918,
                3844.688148985763,
                3892.179804142279,
                3939.684061309631,
                3992.7908662105892,
                4046.0964639926615,
                4094.048026768339,
                4147.715225077025,
                4201.803328840861,
                4250.415130590165,
                4304.400570488394,
                4358.891311144112,
                4413.626381238448,
                4468.247903671312,
                4523.200949785816,
                4584.385109100807,
                4646.338108408218,
                4708.794686519528,
                4771.600484347028,
                4834.854981582876,
                4904.5553996394865,
                4974.441512229322,
                5045.491312421904,
                5130.717507368683,
                5223.421347621407,
                5317.079138212701,
                5418.7071099135665,
                5535.597836296701,
                5667.909048600521,
                5808.1872217746295,
                5956.457881201861,
                6127.4753815696895,
                6314.274205611619,
                6509.692128073985,
                6736.107119503475,
                7008.663641495503,
                7312.530456313335,
                7655.887319816534,
                8100.341736238419,
                8973.536714095795,
                10144.613928413162,
                11315.69114273053,
                12486.768357047898,
                13657.845571365266,
                14828.922785682633,
                16000,
            ]
        )

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if norm == 1:
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]

        # Only check weights if f_mel[0] is positive
        if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
            # This means we have an empty channel somewhere
            warnings.warn(
                "Empty filters detected in mel frequency basis. "
                "Some channels will produce empty responses. "
                "Try increasing your sampling rate (and fmax) or "
                "reducing n_mels."
            )

        return weights


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

        self.logmel_extractor = CustomLogmelFilterBank(
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

        self.bn0 = nn.BatchNorm2d(args_spec.n_mels)

        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=in_channels
        )

        final_in_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(layer, pooling_name)(**args_pooling)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(final_in_features, n_classes)

    def forward(self, x, is_train=True):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if is_train:
        #     x = self.spec_augmenter(x)

        x = x.contiguous().transpose(2, 3)

        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(len(x), -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
