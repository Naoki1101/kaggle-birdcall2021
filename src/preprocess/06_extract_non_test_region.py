import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.append("./src")
import const


def check_test_region(df: pd.DataFrame) -> Dict:
    df["flag1"] = df["longitude"] > -150
    df["flag2"] = df["longitude"] <= -20
    df["flag3"] = df["latitude"] > -20
    df["flag4"] = df["latitude"] <= 60

    df["flag"] = df["flag1"] * df["flag2"] * df["flag3"] * df["flag4"]
    other_regions_df = df[~df["flag"]].reset_index(drop=True)

    count_other_regions = dict(other_regions_df["primary_label"].value_counts())

    return count_other_regions


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train_metadata.csv")

    train_df["flag1"] = train_df["longitude"] > -150
    train_df["flag2"] = train_df["longitude"] <= -20
    train_df["flag3"] = train_df["latitude"] > -20
    train_df["flag4"] = train_df["latitude"] <= 60

    train_df["flag"] = (
        train_df["flag1"] * train_df["flag2"] * train_df["flag3"] * train_df["flag4"]
    )

    non_test_region_idx = train_df[~train_df["flag"]].index

    np.save(const.PROCESSED_DATA_DIR / "non_test_region_idx.npy", non_test_region_idx)


if __name__ == "__main__":
    main()
