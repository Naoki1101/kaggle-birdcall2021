import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
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

    position_dict = {}
    train_distance_array = np.zeros((len(train_df), len(const.BIRD_CODE) - 1))

    gp = train_df.groupby("primary_label")
    for bird, bird_df in gp:
        position_dict[bird] = bird_df[["latitude", "longitude"]].values

    for bird, position_array in tqdm(position_dict.items()):
        train_all_distance = calc_distance(
            train_df[["latitude", "longitude"]].values, position_array
        )
        train_all_distance = np.where(train_all_distance == 0, 1e5, train_all_distance)

        train_distance_array[:, const.BIRD_CODE[bird]] = np.min(
            train_all_distance, axis=1
        )

    np.save(
        const.PROCESSED_DATA_DIR / "distance_array_from_min_position.npy",
        train_distance_array,
    )

    # valid
    valid_position_array = pd.DataFrame.from_dict(const.POS_DICT, orient="index").values

    valid_distance_array = np.zeros(
        (len(valid_position_array), len(const.BIRD_CODE) - 1)
    )

    for bird, position_array in tqdm(position_dict.items()):
        valid_all_distance = calc_distance(valid_position_array, position_array)

        valid_distance_array[:, const.BIRD_CODE[bird]] = np.min(
            valid_all_distance, axis=1
        )

    np.save(
        const.PROCESSED_DATA_DIR / "valid_distance_array_from_min_position.npy",
        valid_distance_array,
    )


if __name__ == "__main__":
    main()
