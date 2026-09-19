#!/usr/bin/env python
"""Print the number of expanded crop records a split partition yields.

`scripts/dental/train_dental.sh` uses this to derive

    train.OFFICIAL_EPOCH_LENGTH = TRAIN_RECORDS / (batch_size_per_gpu * num_gpus)

which is the schedule knob that keeps the DINOv2 run comparable to the MAE /
Swin-MAE / SimCLR runs. The count is a pure function of
``(data_root, split_json, split, categories)``, so it MUST be recomputed whenever
the public split changes. It used to be hardcoded to 68606, which is only correct
for the historical own-collectors-only split; a wrong value silently rescales the
whole training schedule instead of failing loudly.

The expansion itself is not reimplemented here -- it is the exact
``dinov2.data.datasets.dental.build_records`` the SSL dataset uses, so the number
can never drift from what the dataloader iterates over.

Usage:
    python scripts/dental/count_train_records.py \
        --root .datasets/intraoral \
        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json \
        [--split train] [--categories full,tooth,sextant,group]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# baselines/dinov2  (this file lives in baselines/dinov2/scripts/dental/)
DINO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="dataset root (data_root)")
    parser.add_argument("--split_json", required=True, help="case-level split JSON")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="partition of --split_json to expand (default: train)")
    parser.add_argument("--categories", default="full,tooth,sextant,group")
    args = parser.parse_args(argv)

    if str(DINO_ROOT) not in sys.path:
        sys.path.insert(0, str(DINO_ROOT))

    from dinov2.data.datasets.dental import build_records  # noqa: E402

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    records = build_records(args.root, args.split_json, args.split, categories)
    print(len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
