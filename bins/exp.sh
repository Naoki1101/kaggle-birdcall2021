#!/bin/bash

cd ../
# python -m experiments.exp_001 -m -c "First Submit"
# python -m experiments.exp_002 -m -c "legacy_seresnext50_32x4d"
# python -m experiments.exp_003 -m -c "th=0.7 -> 0.65"
# python -m experiments.exp_004 -m -c "use secondary_labels"
# python -m experiments.exp_005 -m -c "mixup"
# python -m experiments.exp_006 -m -c "train only first 80% audio"
# python -m experiments.exp_007 -m -c "add rain noise"
# python -m experiments.exp_008 -m -c "add post-process"
# python -m experiments.exp_009 -m -c "add bus noise"
# python -m experiments.exp_010 -m -c "efficientnet_b0"
# python -m experiments.exp_011 -m -c "efficientnet_b3"
# python -m experiments.exp_012 -m -c "train only first 5[sec] and last 5[sec]"
# python -m experiments.exp_013 -m -c "3tta random shift 2.5[sec]"
# python -m experiments.exp_014 -m -c "efficientnet_b4"
# python -m experiments.exp_015 -m -c "legacy_seresnext50_32x4d, epoch=60"
# python -m experiments.exp_016 -m -c "resnest50d"
# python -m experiments.exp_017 -m -c "resnest50d_4s2x40d"
# python -m experiments.exp_018 -m -c "legacy_seresnext50_32x4d, nocall noise: 0.5 ~ 1.01"
# python -m experiments.exp_019 -m -c "legacy_seresnext50_32x4d, nocall noise: 0.25 ~ 1.01"
# python -m experiments.exp_020 -m -c "use latitude and longitude"
# python -m experiments.exp_021 -m -c "use latitude and longitude, modified"
# python -m experiments.exp_022 -m -c "use latitude and longitude, drop non_test_birds_idx"
# python -m experiments.exp_023 -m -c "use latitude and longitude, drop non_test_region_idx"
# python -m experiments.exp_024 -m -c "use distance from mean position"
# python -m experiments.exp_025 -m -c "use distance from min position"
# python -m experiments.exp_026 -m -c "use audiomentations"
# python -m experiments.exp_027 -m -c "bus_noise and Gain"

# python -m experiments.exp_028 -m -c "torchlibrosa"
# python -m experiments.exp_029 -m -c "AddGaussianNoise"
# python -m experiments.exp_030 -m -c "AddGaussianNoise, add bus noise"
# python -m experiments.exp_031 -m -c "densenet121"
# python -m experiments.exp_032 -m -c "remove SpecAugmentation"
# python -m experiments.exp_033 -m -c "remove Gain"
# python -m experiments.exp_034 -m -c "remove water noise"
# python -m experiments.exp_035 -m -c "ReduceLROnPlateau"
# python -m experiments.exp_036 -m -c "tf_efficientnet_b3_ns"
# python -m experiments.exp_037 -m -c "efficientnet_b3"
# python -m experiments.exp_038 -m -c "SmoothBCEwLogits, smoothing=0.1"
# python -m experiments.exp_039 -m -c "SGD"
# python -m experiments.exp_040 -m -c "seresnext50_32x4d"
# python -m experiments.exp_041 -m -c "seresnext50_32x4d, AddGaussianSNR"
# python -m experiments.exp_042 -m -c "seresnext50_32x4d, AddGaussianSNR, modified_mixup"
# python -m experiments.exp_043 -m -c "seresnext50_32x4d, Gain, AddGaussianSNR, modified_mixup"
# python -m experiments.exp_044 -m -c "seresnext50_32x4d, Gain, AddGaussianSNR, modified_mixup, epoch=90"
# python -m experiments.exp_045 -m -c "seresnext50_32x4d, Gain, AddGaussianSNR, modified_mixup, drop low_rating"
# python -m experiments.exp_046 -m -c "seresnext50_32x4d, Gain, AddGaussianSNR, modified_mixup, contain nocall"
# python -m experiments.exp_047 -m -c "modified exp_045"
# python -m experiments.exp_048 -m -c "noise louder"
# python -m experiments.exp_049 -m -c "copy exp_043, remove train_noise"
# python -m experiments.exp_050 -m -c "copy exp_043, last 5s"
# python -m experiments.exp_051 -m -c "copy exp_043, add Shift"
# python -m experiments.exp_052 -m -c "copy exp_043, add Normalize"
# python -m experiments.exp_053 -m -c "copy exp_043, mixup r < 0.8"
# python -m experiments.exp_054 -m -c "copy exp_043, add non_north_america_idx"   # for SSW and SNE
# python -m experiments.exp_055 -m -c "copy exp_043, add non_south_america_idx"   # for COR and COL
# python -m experiments.exp_056 -m -c "copy exp_043, efficientnet_b3, add COR, SSW score"
# python -m experiments.exp_057 -m -c "copy exp_043, efficientnet_b0, add COR, SSW score"
# python -m experiments.exp_058 -m -c "copy exp_043, use CustomLogmelFilterBank"
# python -m experiments.exp_059 -m -c "copy exp_043, nocall classfication"
# python -m experiments.exp_060 -m -c "copy exp_043, sample 7[sec]"
# python -m experiments.exp_061 -m -c "copy exp_060, modified CV score"
# python -m experiments.exp_062 -m -c "copy exp_061, sample 8[sec]"
# python -m experiments.exp_063 -m -c "copy exp_043, modified CV score, n_mels=156"
# python -m experiments.exp_064 -m -c "copy exp_056, efficientnet_b4, modified CV score"
# python -m experiments.exp_065 -m -c "copy exp_064, efficientnet_b1"
# python -m experiments.exp_066 -m -c "copy exp_064, efficientnet_b2"
# python -m experiments.exp_067 -m -c "copy exp_064, seresnext50_32x4d, nocall&pink noise"
# python -m experiments.exp_068 -m -c "copy exp_067, water&pink noise"
# python -m experiments.exp_069 -m -c "copy exp_067, nocall&water noise, fmin=50, fmax=8000, hop_length=256"
# python -m experiments.exp_070 -m -c "copy exp_059, update"
# python -m experiments.exp_071 -m -c "copy exp_069, fmin=20, fmax=16000, hop_length=256"
# python -m experiments.exp_072 -m -c "copy exp_069, fmin=20, fmax=8000, hop_length=512"
# python -m experiments.exp_073 -m -c "copy exp_069, fmin=50, fmax=16000, hop_length=512"
# python -m experiments.exp_074 -m -c "copy exp_043, StratifiedKFold"
# python -m experiments.exp_075 -m -c "copy exp_073, fmin=20, original smoothing"
python -m experiments.exp_076 -m -c "copy exp_075, smoothing 3nn birds"
python -m experiments.exp_077 -m -c "copy exp_076, smoothing 5nn birds"
python -m experiments.exp_078 -m -c "copy exp_076, smoothing 10nn birds"