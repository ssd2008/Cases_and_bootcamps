"""Применить обученную модель к тестовым аудио и сохранить NPZ и CSV.

NPZ содержит idx: строковый массив (N,) и score: числовой массив (N,).
Чем выше score, тем сильнее модель склоняется к синтетической речи.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Каталог тестовых WAV/FLAC")
    parser.add_argument("--sample", type=Path,
                        help="CSV с idx; определяет точный порядок ответов")
    parser.add_argument("--output", type=Path,
                        default=Path("submission_seed_639320754_639320755.npz"),
                        help="Путь итогового .npz; рядом сохраняются .csv и .json")
    parser.add_argument("--models", type=Path,
                        default=PACKAGE_ROOT / "models/research_final_r10/FL_REPEAT",
                        help="Каталог ensemble.json, нормировщика и двух весов")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                        default="auto")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output должен иметь расширение .npz")
    if args.batch_size < 1:
        parser.error("--batch-size должен быть положительным")
    if not args.input.is_dir():
        parser.error("--input должен указывать на каталог аудио")
    if not (args.models / "ensemble.json").is_file():
        parser.error("В --models отсутствует ensemble.json")

    import numpy as np
    from predict_ensemble import predict

    # Временный рецепт позволяет использовать и исходные, и новые веса.
    recipe = {
        "name": "FL_REPEAT",
        "research_only": False,
        "equal_mean": True,
        "components": [{"id": "BEST100", "model_dir": str(args.models.resolve()),
                        "weight": 1.0}],
    }
    output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="rucode_recipe_") as tmp:
        recipe_path = Path(tmp) / "recipe.json"
        recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
        predict(args.input.resolve(), args.sample.resolve() if args.sample else None,
                recipe_path, output.with_suffix(".csv"),
                args.batch_size, args.device)

    # Для сдачи оставляем только документированные массивы, без внутренних частей.
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
        idx, score = data["idx"].copy(), data["score"].copy()
    np.savez_compressed(output, idx=idx, score=score)
    receipt_path = output.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    import hashlib
    receipt["npz"] = {"path": str(output), "keys": ["idx", "score"],
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(f"NPZ: {output} ({len(idx)} записей)")


if __name__ == "__main__":
    main()
