import sys

import pandas as pd

sys.path.append("./src")
import const
from utils import DataHandler

dh = DataHandler()


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train_metadata.csv")
    valid_df = pd.read_csv(const.INPUT_DATA_DIR / "train_soundscape_labels.csv")

    gp = valid_df.groupby("audio_id")

    position_dict = {}
    for audio_id, audio_df in gp:
        unique_birds = audio_df["birds"].unique()

        df = train_df[train_df["primary_label"].isin(unique_birds)]
        if len(df) == 0:
            df = train_df.copy()

        audio_position_dict = {
            "latitude": df["latitude"].mean(),
            "longitude": df["longitude"].mean(),
        }

        position_dict[audio_id] = audio_position_dict

    dh.save(const.INPUT_DATA_DIR / "valid_position.json", position_dict)


if __name__ == "__main__":
    main()
