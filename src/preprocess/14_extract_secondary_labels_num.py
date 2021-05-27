import re
import sys

import numpy as np

sys.path.append("./src")
import const
from utils import DataHandler

dh = DataHandler()


def main():
    train_df = dh.load(const.INPUT_DATA_DIR / "train_metadata.csv")
    train_df["secondary_labels_num"] = train_df["secondary_labels"].apply(
        lambda x: len(re.findall("'", x)) // 2
    )
    many_secondary_labels_idx = train_df[
        train_df["secondary_labels_num"] >= 2
    ].index.values
    np.save(
        const.PROCESSED_DATA_DIR / "many_secondary_labels_idx.npy",
        many_secondary_labels_idx,
    )


if __name__ == "__main__":
    main()
