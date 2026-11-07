"""Проверить запуск FL_REPEAT без конкурсных данных."""

import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


def main():
    root = Path(__file__).resolve().parent
    with TemporaryDirectory(prefix="rucode-demo-") as tmp:
        folder = Path(tmp)
        audio = folder / "audio"
        audio.mkdir()
        time = np.arange(16000) / 16000
        names = ["sweep.wav", "tone.wav"]
        sf.write(audio / names[0], 0.3 * np.sin(2 * np.pi * (180 * time + 80 * time**2)), 16000)
        sf.write(audio / names[1], 0.3 * np.sin(2 * np.pi * 220 * time), 16000)
        sample = folder / "sample.csv"
        sample.write_text("idx\n" + "\n".join(names) + "\n")
        output = folder / "scores.npz"
        subprocess.run(
            [sys.executable, str(root / "predict.py"), "--input", str(audio),
             "--sample", str(sample), "--output", str(output), "--device", "cpu"],
            check=True,
        )
        with np.load(output, allow_pickle=False) as data, output.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
            scores = data["score"]
            if (
                set(data.files) != {"idx", "score"}
                or data["idx"].tolist() != names
                or [row["idx"] for row in rows] != names
                or scores.shape != (len(names),)
                or not np.all((0 <= scores) & (scores <= 1))
                or not np.array_equal(scores, [float(row["score"]) for row in rows])
            ):
                raise RuntimeError("CSV и NPZ должны содержать одинаковые оценки в порядке sample.")
        print("Демо прошло. Оценки на тестовых сигналах:")
        for name, score in zip(names, scores):
            print(f"{name}: {score:.6f}")


if __name__ == "__main__":
    main()
