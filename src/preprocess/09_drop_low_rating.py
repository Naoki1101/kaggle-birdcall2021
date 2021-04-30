import sys

import numpy as np
import pandas as pd

sys.path.append("./src")
import const


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train_metadata.csv")
    low_rating_idx = train_df[train_df["rating"] <= 3.0].index
    np.save(const.PROCESSED_DATA_DIR / "low_rating.npy", low_rating_idx)


if __name__ == "__main__":
    main()
