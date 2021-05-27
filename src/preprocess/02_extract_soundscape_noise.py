import sys

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

sys.path.append("./src")
import const


def extract_noise(df: pd.DataFrame, file_name: str) -> np.array:
    audio_path = const.TRAIN_SOUNDSCAPES_DIR / file_name
    sample, sr = librosa.load(
        audio_path, sr=const.TARGET_SAMPLE_RATE, mono=True, res_type="kaiser_fast"
    )

    noise_list = []
    for idx in range(len(df)):
        s = df.loc[idx, "pre_seconds"] * const.TARGET_SAMPLE_RATE
        e = df.loc[idx, "seconds"] * const.TARGET_SAMPLE_RATE
        noise_list.append(sample[int(s) : int(e)])

    noise = np.concatenate(noise_list)

    return noise


def main():
    df = pd.read_csv(const.INPUT_DATA_DIR / "train_soundscape_labels.csv")
    df["file_id"] = df["audio_id"].astype(str) + "_" + df["site"]
    df["file_name"] = df["file_id"].map(const.TRAIN_SOUNDSCAPES_CODE)
    df = df[df["audio_id"].isin([7019, 7954, 14473])].reset_index(drop=True)

    all_noise_list = []
    file_gp = df.groupby("file_name")
    for file_name, file_df in file_gp:
        file_df["pre_seconds"] = [0] + file_df["seconds"].shift(1).tolist()[1:]
        noise_df = file_df[file_df["birds"] == "nocall"].reset_index(drop=True)

        if len(noise_df) > 0:
            file_noise = extract_noise(noise_df, file_name)
            all_noise_list.append(file_noise)

    noise = np.concatenate(all_noise_list)

    const.NOISE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(
        const.NOISE_AUDIO_DIR / "train_soundscape_nocall.wav",
        noise,
        samplerate=const.TARGET_SAMPLE_RATE,
    )


if __name__ == "__main__":
    main()
