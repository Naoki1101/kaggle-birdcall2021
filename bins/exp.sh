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
# python -m experiments.exp_076 -m -c "copy exp_075, smoothing 3nn birds"
# python -m experiments.exp_077 -m -c "copy exp_076, smoothing 5nn birds"
# python -m experiments.exp_078 -m -c "copy exp_076, smoothing 10nn birds"
# python -m experiments.exp_079 -m -c "copy exp_070, modified"
# python -m experiments.exp_080 -m -c "copy exp_079, nocall & pink & water & motorcycle noise"
# python -m experiments.exp_081 -m -c "copy exp_073, fmin=20, noise coef ~ 50"
# python -m experiments.exp_082 -m -c "copy exp_081, noise coef ~ 100"
# python -m experiments.exp_083 -m -c "copy exp_079, noise coef ~ 50"
# python -m experiments.exp_084 -m -c "copy exp_079, noise coef ~ 10, remove Gain, add water noise"
# python -m experiments.exp_085 -m -c "copy exp_081, PitchShift"   # 時間かかるからskip
# python -m experiments.exp_086 -m -c "copy exp_081, resnet101d"
# python -m experiments.exp_087 -m -c "copy exp_081, 3mixup"
# python -m experiments.exp_088 -m -c "copy exp_081, Normalize"
# python -m experiments.exp_089 -m -c "copy exp_081, resnext50_32x4d"
# python -m experiments.exp_090 -m -c "copy exp_081, coef ~ 10, train first 5[sec] and last 5[sec]"
# python -m experiments.exp_091 -m -c "copy exp_081, coef ~ 10, nocall&speaking noise"
# python -m experiments.exp_092 -m -c "copy exp_091, r >= 0.9 then y = 0 ~ 5[sec] + 5 ~ 10[sec]"
# python -m experiments.exp_093 -m -c "copy exp_092, nocall&crowd noise"
# python -m experiments.exp_094 -m -c "copy exp_092, nfnet_f0"   # 途中でエラーになるからとりあえずskipした
# python -m experiments.exp_095 -m -c "copy exp_093, crowd noise -> contains crowd noise"
# python -m experiments.exp_096 -m -c "copy exp_092, r >= 0.7 then y = 0 ~ 5[sec] + 5 ~ 10[sec]"
# python -m experiments.exp_097 -m -c "copy exp_092, only random noise"
# python -m experiments.exp_098 -m -c "copy exp_097, remove bn0"
# python -m experiments.exp_099 -m -c "copy exp_079, only random noise"
# python -m experiments.exp_100 -m -c "copy exp_097, Shift, AddGaussianSNR=1.0"
# python -m experiments.exp_101 -m -c "copy exp_100, Shift, AddGaussianSNR=0.5"
# python -m experiments.exp_102 -m -c "copy exp_004, librosa, Gain, AddGaussianSNR, random noise, mixup"
# python -m experiments.exp_103 -m -c "copy exp_091, nocall&speaking&water noise"
# python -m experiments.exp_104 -m -c "copy exp_097, top_db=80.0"
# python -m experiments.exp_105 -m -c "copy exp_104, n_mels=256"
# python -m experiments.exp_106 -m -c "copy exp_104, BCEFocalLoss"
# python -m experiments.exp_107 -m -c "copy exp_105, resize(224, 224), PReLU"
# python -m experiments.exp_108 -m -c "copy exp_104, n_fft=4096"
# python -m experiments.exp_109 -m -c "copy exp_104, SpecAugmentation"
# python -m experiments.exp_110 -m -c "copy exp_097, n_fft=4096, Shift, tob_db=None"
# python -m experiments.exp_111 -m -c "copy exp_110, snr"
# python -m experiments.exp_112 -m -c "copy exp_111, pink&water&speaking noise"
# python -m experiments.exp_113 -m -c "copy exp_112, epochs=60, T_max=20"
# python -m experiments.exp_114 -m -c "copy exp_112, 7sec"
# python -m experiments.exp_115 -m -c "copy exp_112, efficientnet_b0"
# python -m experiments.exp_116 -m -c "copy exp_114, efficientnet_b0"
# python -m experiments.exp_117 -m -c "copy exp_112, efficientnet_b3"
# python -m experiments.exp_118 -m -c "copy exp_114, efficientnet_b3"
# python -m experiments.exp_119 -m -c "copy exp_112, r >= 0.9 then y = 0 ~ 5[sec] + random crop"
# python -m experiments.exp_120 -m -c "copy exp_119, pos_weight"
# python -m experiments.exp_121 -m -c "copy exp_119, SpecAugmentation"
# python -m experiments.exp_122 -m -c "copy exp_119, efficientnet_b4"
# python -m experiments.exp_123 -m -c "copy exp_119, efficientnet_b2"
# python -m experiments.exp_124 -m -c "copy exp_119, th=0.45"
# python -m experiments.exp_125 -m -c "copy exp_121, add crowd noise"
# python -m experiments.exp_126 -m -c "copy exp_121, add rain noise"
# python -m experiments.exp_127 -m -c "copy exp_121, mixup=False"
# python -m experiments.exp_128 -m -c "copy exp_121, change params of SpecAugmentation"
# python -m experiments.exp_129 -m -c "copy exp_121, GaussianBlur"
# python -m experiments.exp_130 -m -c "copy exp_129, duplicate samples with less than 50 samples"
# python -m experiments.exp_131 -m -c "copy exp_129, 7sec"
# python -m experiments.exp_132 -m -c "copy exp_129, efficientnet_b2"
# python -m experiments.exp_133 -m -c "copy exp_131, efficientnet_b2, 7sec"
# python -m experiments.exp_134 -m -c "copy exp_129, efficientnet_b3"
# python -m experiments.exp_135 -m -c "copy exp_131, efficientnet_b3, 7sec"
# python -m experiments.exp_136 -m -c "copy exp_132, rm GaussianBlur"
# python -m experiments.exp_137 -m -c "copy exp_133, rm GaussianBlur"
# python -m experiments.exp_138 -m -c "copy exp_134, rm GaussianBlur"
# python -m experiments.exp_139 -m -c "copy exp_135, rm GaussianBlur"
# python -m experiments.exp_140 -m -c "copy exp_119, vit_base_resnet50d_224"   # 全く学習してくれない
# python -m experiments.exp_141 -m -c "copy exp_129, resnest50d"

# python -m experiments.exp_142 -m -c "copy exp_129, fix mixup"
# python -m experiments.exp_143 -m -c "copy exp_142, rm GaussianBlur"
# python -m experiments.exp_144 -m -c "copy exp_143, drop many_secondary_labels_idx"
# python -m experiments.exp_145 -m -c "copy exp_144, rm SpecAugmentation"
# python -m experiments.exp_146 -m -c "copy exp_144, 7sec"
# python -m experiments.exp_147 -m -c "copy exp_144, change SpecAugmentation"
# python -m experiments.exp_148 -m -c "copy exp_147, 7sec"
# python -m experiments.exp_149 -m -c "copy exp_144, efficientnet_b2"
# python -m experiments.exp_150 -m -c "copy exp_144, efficientnet_b3"
# python -m experiments.exp_151 -m -c "copy exp_144, rm mixup"
# python -m experiments.exp_152 -m -c "copy exp_149, rm mixup"
# python -m experiments.exp_153 -m -c "copy exp_150, rm mixup"
# python -m experiments.exp_154 -m -c "copy exp_146, efficientnet_b2"
# python -m experiments.exp_155 -m -c "copy exp_146, efficientnet_b3"
# python -m experiments.exp_156 -m -c "copy exp_146, rm mixup"
# python -m experiments.exp_157 -m -c "copy exp_154, rm mixup"
# python -m experiments.exp_158 -m -c "copy exp_155, rm mixup"
# python -m experiments.exp_159 -m -c "copy exp_143, secondary_labels = 0.3"
# python -m experiments.exp_160 -m -c "copy exp_159, efficientnet_b2"
# python -m experiments.exp_161 -m -c "copy exp_159, efficientnet_b3"

# python -m experiments.exp_162 -m -c "copy exp_159, Save weight every time CV is improved, minmax"
# python -m experiments.exp_163 -m -c "copy exp_162, BN"
# python -m experiments.exp_164 -m -c "copy exp_163, rm mixup"
# python -m experiments.exp_165 -m -c "copy exp_163, 7sec"
# python -m experiments.exp_166 -m -c "copy exp_165, rm mixup"
# python -m experiments.exp_167 -m -c "copy exp_163, efficientnet_b3"
# python -m experiments.exp_168 -m -c "copy exp_164, efficientnet_b3"
# python -m experiments.exp_169 -m -c "copy exp_165, efficientnet_b3"
# python -m experiments.exp_170 -m -c "copy exp_166, efficientnet_b3"
# python -m experiments.exp_171 -m -c "copy exp_167, efficientnet_b2"
# python -m experiments.exp_172 -m -c "copy exp_168, efficientnet_b2"
# python -m experiments.exp_173 -m -c "copy exp_169, efficientnet_b2"
# python -m experiments.exp_174 -m -c "copy exp_170, efficientnet_b2"
# python -m experiments.exp_175 -m -c "copy exp_164, optim threshhold every epoch"
# python -m experiments.exp_176 -m -c "copy exp_099, fix bug, pink&water&speaking noise, use rms"
# python -m experiments.exp_177 -m -c "copy exp_163, epochs=50"
# python -m experiments.exp_178 -m -c "copy exp_165, epochs=50"
# python -m experiments.exp_179 -m -c "copy exp_167, epochs=50"
# python -m experiments.exp_180 -m -c "copy exp_169, epochs=50"
# python -m experiments.exp_181 -m -c "copy exp_171, epochs=50"
# python -m experiments.exp_182 -m -c "copy exp_173, epochs=50"
# python -m experiments.exp_183 -m -c "copy exp_165, efficientnetv2_s"   # 3epochまで学習してくれないからskip
# python -m experiments.exp_184 -m -c "copy exp_165, tf_efficientnetv2_b3"
python -m experiments.exp_185 -m -c "copy exp_163, tf_efficientnetv2_b3"
python -m experiments.exp_186 -m -c "copy exp_164, tf_efficientnetv2_b3"
python -m experiments.exp_187 -m -c "copy exp_166, tf_efficientnetv2_b3"

# python -m experiments.exp_999 -m -c "test"



############################
# Final Model #
############################
## resnext50_32x4d, 5s, SpecAugmentation
# 

## resnext50_32x4d, 7s, SpecAugmentation
# 

## resnext50_32x4d, 5s, mixup, SpecAugmentation
# python -m experiments.exp_163 -m -c "copy exp_162, BN"

## resnext50_32x4d, 7s, mixup, SpecAugmentation
# 


## efficientnet_b2, 5s, SpecAugmentation
# 

## efficientnet_b2, 7s, SpecAugmentation
# 

## efficientnet_b2, 5s, mixup, SpecAugmentation
# 

## efficientnet_b2, 7s, mixup, SpecAugmentation
# 


## efficientnet_b3, 5s, SpecAugmentation
# 

## efficientnet_b3, 7s, SpecAugmentation
# 

## efficientnet_b3, 5s, mixup, SpecAugmentation
# 

## efficientnet_b3, 7s, mixup, SpecAugmentation
# 
