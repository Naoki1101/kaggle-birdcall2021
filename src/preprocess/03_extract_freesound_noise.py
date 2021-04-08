import sys

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

sys.path.append("./src")
import const


def extract_noise(df: pd.DataFrame) -> np.array:
    noise_list = []
    for idx in df.index:
        fname = df.loc[idx, "fname"]
        noise, sr = librosa.load(const.FREESOUND_DIR / fname, sr=44_100)
        noise_list.append(noise)

    noise = np.concatenate(noise_list)

    return noise


def save_noise(noise: np.array, file_name) -> None:
    sf.write(
        const.NOISE_AUDIO_DIR / f"freesound_{file_name}.wav",
        noise,
        samplerate=const.TARGET_SAMPLE_RATE,
    )


def main() -> None:
    df = pd.read_csv(const.FREESOUND_DIR / "train_curated.csv")

    audio_length_list = []
    for idx in df.index:
        audio_fname = df.loc[idx, "fname"]
        audio, sr = librosa.load(const.FREESOUND_DIR / audio_fname, sr=44_100)
        audio_length = len(audio) // sr
        audio_length_list.append(audio_length)

    df["length"] = audio_length_list
    df = df[df["length"] >= 3].reset_index(drop=True)

    water_noise_df = df[
        df["labels"].isin(["Trickle_and_dribble", "Fill_(with_liquid)", "Drip"])
    ].reset_index(drop=True)
    bus_noise_df = df[df["labels"].isin(["Bus"])].reset_index(drop=True)
    walk_noise_df = df[df["labels"].isin(["Walk_and_footsteps"])].reset_index(drop=True)
    rain_noise_df = df[df["labels"].isin(["Raindrop"])].reset_index(drop=True)
    motorcycle_noise_df = df[df["labels"].isin(["Motorcycle"])].reset_index(drop=True)

    water_noise = extract_noise(water_noise_df)
    bus_noise = extract_noise(bus_noise_df)
    walk_noise = extract_noise(walk_noise_df)
    rain_noise = extract_noise(rain_noise_df)
    motorcycle_noise = extract_noise(motorcycle_noise_df)

    save_noise(water_noise, "water_noise")
    save_noise(bus_noise, "bus_noise")
    save_noise(walk_noise, "walk_noise")
    save_noise(rain_noise, "rain_noise")
    save_noise(motorcycle_noise, "motorcycle_noise")


if __name__ == "__main__":
    main()
