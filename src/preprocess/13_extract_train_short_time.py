import sys

import pandas as pd
from tqdm import tqdm
import librosa

sys.path.append("./src")
import const
from utils import DataHandler

dh = DataHandler()


def extract_length(df: pd.DataFrame, sr: float):
    wave_length_list = []

    for idx in tqdm(df.index):
        pl = df.loc[idx, "primary_label"]
        fn = df.loc[idx, "filename"]

        file_path = const.TRAIN_AUDIO_DIR / pl / fn
        y, sr = librosa.load(file_path, sr=const.TARGET_SAMPLE_RATE)

        wave_length_list.append(len(y))

    return wave_length_list


def main():
    train_df = dh.load(const.INPUT_DATA_DIR / "train_metadata.csv")
    wave_length_list = extract_length(train_df, sr=const.TARGET_SAMPLE_RATE)
    wave_length_df = pd.DataFrame(wave_length_list, columns=["wave_length"])
    dh.save(const.PROCESSED_DATA_DIR / "train_short_wave_length.csv", wave_length_df)


if __name__ == "__main__":
    main()
