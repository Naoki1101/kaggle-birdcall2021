import sys
import warnings

import pandas as pd
import librosa
import soundfile as sf
from joblib import Parallel, delayed

sys.path.append("./src")
import const

warnings.simplefilter("ignore")

NUM_THREAD = 4  # for joblib.Parallel


# https://www.kaggle.com/c/birdsong-recognition/discussion/159943
def resample(primary_label: str, filename: str, target_sr: int):
    audio_dir = const.TRAIN_AUDIO_DIR
    resample_dir = const.TRAIN_RESAMPLED_AUDIO_DIR
    ebird_dir = resample_dir / primary_label

    try:
        y, _ = librosa.load(
            str(audio_dir / primary_label / filename),
            sr=target_sr,
            mono=True,
            res_type="kaiser_fast",
        )

        filename = filename.replace(".ogg", ".wav")
        sf.write(ebird_dir / filename, y, samplerate=target_sr)
        return "OK"
    except Exception as e:
        with open(resample_dir / "skipped.txt", "a") as f:
            file_path = str(audio_dir / primary_label / filename)
            f.write(file_path + "\n")
        return str(e)


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train_metadata.csv")
    train_audio_infos = train_df[["primary_label", "filename"]].values.tolist()

    const.TRAIN_RESAMPLED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for primary_label in train_df["primary_label"].unique():
        ebird_dir = const.TRAIN_RESAMPLED_AUDIO_DIR / primary_label
        ebird_dir.mkdir(exist_ok=True)

    _ = Parallel(n_jobs=NUM_THREAD, verbose=1)(
        delayed(resample)(primary_label, file_name, const.TARGET_SAMPLE_RATE)
        for primary_label, file_name in train_audio_infos
    )

    train_df["resampled_sampling_rate"] = const.TARGET_SAMPLE_RATE
    train_df["resampled_filename"] = train_df["filename"].map(
        lambda x: x.replace(".ogg", ".wav")
    )
    train_df["resampled_channels"] = "1 (mono)"

    train_df[
        [
            "primary_label",
            "filename",
            "resampled_sampling_rate",
            "resampled_filename",
            "resampled_channels",
        ]
    ].to_csv(const.INPUT_DATA_DIR / "train_mod.csv", index=False)


if __name__ == "__main__":
    main()
