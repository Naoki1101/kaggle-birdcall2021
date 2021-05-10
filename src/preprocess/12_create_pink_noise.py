import sys

import acoustics
import soundfile as sf

sys.path.append("./src")
import const


def main():
    pink_noise = acoustics.generator.pink(120 * const.TARGET_SAMPLE_RATE)
    sf.write(
        const.NOISE_AUDIO_DIR / "pink_noise.wav",
        pink_noise,
        samplerate=const.TARGET_SAMPLE_RATE,
    )


if __name__ == "__main__":
    main()
