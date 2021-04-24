import sys

import numpy as np
import pandas as pd
from scipy.spatial import distance

sys.path.append("./src")
import const
from utils import DataHandler

dh = DataHandler()


def calc_distance(
    matrix1: np.array, matrix2: np.array, method: str = "euclidean"
) -> np.array:
    concat_matrix = np.concatenate([matrix1, matrix2], axis=0)
    dis = distance.cdist(concat_matrix, concat_matrix, metric="euclidean")
    return dis[: len(matrix1), -len(matrix2) :]


def main():
    # train
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train_metadata.csv")

    mean_latitude = train_df.groupby("primary_label")["latitude"].mean()
    mean_longitude = train_df.groupby("primary_label")["longitude"].mean()
    mean_position_array = pd.concat([mean_latitude, mean_longitude], axis=1).values

    train_position_array = train_df[["latitude", "longitude"]].values
    train_distance_array = calc_distance(train_position_array, mean_position_array)

    np.save(
        const.PROCESSED_DATA_DIR / "distance_array_from_mean_position.npy",
        train_distance_array,
    )

    # valid
    valid_position_array = pd.DataFrame.from_dict(const.POS_DICT, orient="index").values
    valid_distance_array = calc_distance(valid_position_array, mean_position_array)

    np.save(
        const.PROCESSED_DATA_DIR / "valid_distance_array_from_mean_position.npy",
        valid_distance_array,
    )


if __name__ == "__main__":
    main()
