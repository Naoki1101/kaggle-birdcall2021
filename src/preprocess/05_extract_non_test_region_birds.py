import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.append("./src")
import const


def count_other_region(df: pd.DataFrame) -> Dict:
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

    count_df = train_df["primary_label"].value_counts().reset_index()
    count_df.columns = ["primary_label", "count_all_region"]

    other_ce = count_other_region(train_df)
    count_df["count_non_test_regions"] = count_df["primary_label"].map(other_ce)

    count_df["percent_other_regions"] = (
        count_df["count_non_test_regions"] / count_df["count_all_region"]
    )

    drop_birds = count_df[count_df["percent_other_regions"] >= 0.8][
        "primary_label"
    ].values
    non_test_birds_idx = train_df[train_df["primary_label"].isin(drop_birds)].index

    np.save(const.PROCESSED_DATA_DIR / "non_test_birds_idx.npy", non_test_birds_idx)


if __name__ == "__main__":
    main()
