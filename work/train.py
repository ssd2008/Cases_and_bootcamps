"""Обучить FL_REPEAT на предоставленном train и сохранить две модели.

Пример: python train.py --train /path/to/train --device auto
Новые веса сохраняются отдельно от включённых в пакет исходных весов.
"""

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True,
                        help="Каталог с targets.txt и подкаталогом flac/")
    parser.add_argument("--out", type=Path,
                        default=PACKAGE_ROOT / "models/retrained/FL_REPEAT",
                        help="Каталог новых весов внутри models/retrained/")
    parser.add_argument("--draws-per-class", type=int, default=3000,
                        help="Число примеров каждого класса за эпоху (3000 в исходном рецепте)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                        default="auto")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers должен быть положительным")
    if args.draws_per_class < 1:
        parser.error("--draws-per-class должен быть положительным")
    args.recipe = "FL_REPEAT"
    from train_final import main as train
    train(args)


if __name__ == "__main__":
    main()
