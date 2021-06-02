#!/bin/bash

mkdir -p data
mkdir -p data/input
mkdir -p data/freesound
mkdir -p data/processed
mkdir -p logs

# pip
pip install -U pip
pip install -r requirements.txt

## kaggle api
touch ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
export KAGGLE_USERNAME=xxxxxxxxxxxxxx
export KAGGLE_KEY=xxxxxxxxxxxxxx


# download birdsong-recognition
cd ./data/input/
kaggle competitions download -c birdclef-2021 -p .
unzip ./*.zip

# download freesound-audio-tagging-2019
cd ../freesound
kaggle competitions download -c freesound-audio-tagging-2019 -f train_curated.csv -p .
kaggle competitions download -c freesound-audio-tagging-2019 -f train_curated.zip -p .
unzip train_curated.zip
rm -rf train_curated.zip