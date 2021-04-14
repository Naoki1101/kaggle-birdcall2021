#!/bin/bash

cd ../
# python -m experiments.exp_001 -m -c "First Submit"
# python -m experiments.exp_002 -m -c "legacy_seresnext50_32x4d"
# python -m experiments.exp_003 -m -c "th=0.7 -> 0.65"
# python -m experiments.exp_004 -m -c "use secondary_labels"
# python -m experiments.exp_005 -m -c "mixup"
# python -m experiments.exp_006 -m -c "train only first 80% audio"
# python -m experiments.exp_007 -m -c "add rain noise"   # 後回し
# python -m experiments.exp_008 -m -c "add post-process"   # これ回す必要なくないかw？
# python -m experiments.exp_009 -m -c "add bus noise"
# python -m experiments.exp_010 -m -c "efficientnet_b0"
# python -m experiments.exp_011 -m -c "efficientnet_b3"
python -m experiments.exp_012 -m -c "train only first 5[sec] and last 5[sec]"
