# Copyright (c) 2025. Licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Dental SSL dataset for DINOv2 continued self-supervised pretraining (§9.5.2).
#
# Reads the public case-level split JSON produced by the workspace root
# `datasets/split.py` (create_or_load_split) and expands each annotation JSON
# into crop records:
#
#   full    -> one record per view annotation (the full intraoral image)
#   tooth   -> one record per tooth box        (ann["teeth"][].bbox_padded)
#   sextant -> one record per sextant box      (ann["sextants"][].bbox_padded)
#   group   -> dynamic four-tooth groups       (four_tooth_group_candidates +
#              union_and_pad_boxes on member teeth bbox_padded)
#
# Every record carries a stable `sample_key`::
#
#     "{case_id}:{view}:{crop_type}:{label}"
#
# so downstream feature evaluation (the `compare/` package) can align samples
# across models. The geometry/group helpers below mirror the workspace root
# implementations in ``datasets/dataset.py`` and ``datasets/fdi_relation.py``
# on purpose: this file must stay self-contained so the baseline repo can run
# without import-path dependencies on the parent workspace.
#
# The dataset implements the ``ExtendedVisionDataset`` contract used by
# ``dinov2/data/loaders.py::make_dataset``: ``__getitem__`` returns a
# ``(PIL image, dummy target)`` pair and the existing ``DataAugmentationDINO``
# / collate pipeline (``train/train.py::do_train``) performs all multi-crop
# augmentation. Crops other than "full" are re-cut from the full-resolution
# source image (record ``source_image_path`` + ``box_original``) so pixel
# quality matches what downstream feature evaluation will feed the model.

import argparse
import hashlib
import io
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL.Image

from .decoders import DecoderType, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")

PAD_RATIO = 0.1  # must match datasets/dataset.py::_union_and_pad_boxes default

CROP_TYPES = ("full", "tooth", "sextant", "group")

_SPLIT_ALIASES = {
    "TRAIN": "train",
    "train": "train",
    "VAL": "test",
    "val": "test",
    "TEST": "test",
    "test": "test",
}


# --- geometry helpers (vendored from datasets/dataset.py) -----------------


def clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [max(0, min(x1, img_w)), max(0, min(y1, img_h)),
            max(0, min(x2, img_w)), max(0, min(y2, img_h))]


def valid_box(box):
    return box[2] > box[0] and box[3] > box[1]


def union_and_pad_boxes(boxes, img_w, img_h, pad_ratio=PAD_RATIO):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return clamp_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], img_w, img_h)


# --- FDI group helpers (vendored from datasets/fdi_relation.py) -----------

_SEXTANT_TABLE = {
    1: {(1, 3): "S2", (4, 8): "S1"},
    2: {(1, 3): "S2", (4, 8): "S3"},
    3: {(1, 3): "S5", (4, 8): "S4"},
    4: {(1, 3): "S5", (4, 8): "S6"},
}


def parse_fdi(fdi):
    """11 -> (quadrant=1, tooth_index=1)."""
    return fdi // 10, fdi % 10


def sextant_of(fdi):
    q, t = parse_fdi(fdi)
    for (lo, hi), sextant in _SEXTANT_TABLE[q].items():
        if lo <= t <= hi:
            return sextant
    raise ValueError(f"invalid FDI code: {fdi}")


def four_tooth_group_candidates(view, available_fdis):
    """Return strict four-tooth groups whose members are all present."""
    view = str(view).upper()
    if view == "F":
        candidates = [
            (11, 21, 41, 31),
            (11, 12, 41, 42),
            (12, 13, 42, 43),
            (21, 22, 31, 32),
            (22, 23, 32, 33),
        ]
    elif view == "R":
        candidates = [
            (14, 15, 44, 45),
            (15, 16, 45, 46),
            (16, 17, 46, 47),
            (17, 18, 47, 48),
        ]
    elif view == "L":
        candidates = [
            (24, 25, 34, 35),
            (25, 26, 35, 36),
            (26, 27, 36, 37),
            (27, 28, 37, 38),
        ]
    elif view in {"U", "D"}:
        arch = (
            [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
            if view == "U"
            else [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
        )
        candidates = [tuple(arch[i:i + 4]) for i in range(len(arch) - 3)]
    else:
        raise ValueError(f"unsupported intraoral view: {view}")

    available = {int(fdi) for fdi in available_fdis}
    return [group for group in candidates if all(fdi in available for fdi in group)]


# --- split / annotation loading -------------------------------------------

def load_split(split_json):
    split_path = Path(split_json)
    if not split_path.exists():
        raise FileNotFoundError(f"split_json does not exist: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_annotation_paths(split, split_name):
    """Annotation JSON paths for a partition (train/test).

    Returns absolute paths when the split JSON carries an absolute ``data_root``;
    otherwise returns the raw relative paths so the caller can resolve them
    against the data_root it was invoked with (avoids double-prefixing when
    the split JSON stores a relative ``data_root``).
    """
    data_root = split.get("data_root") or ""
    rel_paths = split["annotations"][split_name]
    if data_root and Path(data_root).is_absolute():
        return [str(Path(data_root) / rel_path) for rel_path in rel_paths]
    return [str(Path(rel_path)) for rel_path in rel_paths]


def load_annotation(json_path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def full_image_path(json_path, ann):
    """Full-resolution original image referenced by an annotation JSON."""
    json_path = Path(json_path)
    return json_path.parent.parent.parent / ann["image_path"]


def make_sample_key(case_id, view, crop_type, label):
    return f"{case_id}:{view}:{crop_type}:{label}"


def _build_tooth_records(ann, json_path, case_id, view, partition):
    records = []
    full_path = full_image_path(json_path, ann)
    for raw in ann.get("teeth", []):
        box = clamp_box(raw["bbox_padded"], ann["orig_size"][0], ann["orig_size"][1])
        if not valid_box(box):
            continue
        fdi = int(raw["fdi"])
        records.append({
            "case_id": case_id,
            "view": view,
            "crop_type": "tooth",
            "label": fdi,
            "sample_key": make_sample_key(case_id, view, "tooth", fdi),
            "source_image_path": str(full_path),
            "box_original": [float(v) for v in box],
            "annotation_path": str(json_path),
            "partition": partition,
        })
    return records


def _build_sextant_records(ann, json_path, case_id, view, partition):
    records = []
    full_path = full_image_path(json_path, ann)
    for raw in ann.get("sextants", []):
        box = clamp_box(raw["bbox_padded"], ann["orig_size"][0], ann["orig_size"][1])
        if not valid_box(box):
            continue
        sextant_id = str(raw["id"])
        records.append({
            "case_id": case_id,
            "view": view,
            "crop_type": "sextant",
            "label": sextant_id,
            "sample_key": make_sample_key(case_id, view, "sextant", sextant_id),
            "source_image_path": str(full_path),
            "box_original": [float(v) for v in box],
            "annotation_path": str(json_path),
            "partition": partition,
        })
    return records


def _build_group_records(ann, json_path, case_id, view, partition):
    img_w, img_h = ann["orig_size"]
    tooth_boxes = {}
    for raw in ann.get("teeth", []):
        box = clamp_box(raw["bbox_padded"], img_w, img_h)
        if valid_box(box):
            tooth_boxes[int(raw["fdi"])] = box
    full_path = full_image_path(json_path, ann)
    records = []
    for member_fdi in four_tooth_group_candidates(view, tooth_boxes):
        label = "-".join(str(fdi) for fdi in member_fdi)
        records.append({
            "case_id": case_id,
            "view": view,
            "crop_type": "group",
            "label": label,
            "sample_key": make_sample_key(case_id, view, "group", label),
            "source_image_path": str(full_path),
            "box_original": [float(v) for v in union_and_pad_boxes(
                [tooth_boxes[fdi] for fdi in member_fdi], img_w, img_h)],
            "annotation_path": str(json_path),
            "partition": partition,
        })
    return records


def build_records(data_root, split_json, split, categories):
    """Expand split annotations into crop records for SSL pretraining.

    Args:
        data_root: dataset root; used to resolve the split's relative paths
            when the split JSON lacks an absolute ``data_root`` field.
        split_json: path to the case-level split JSON (create_or_load_split).
        split: partition name ("train" or "test").
        categories: iterable of crop types, subset of ("full","tooth","sextant","group").
    """
    split_data = load_split(split_json)
    data_root = str(Path(data_root).resolve())
    split_data = dict(split_data)
    if not split_data.get("data_root"):
        split_data["data_root"] = data_root

    categories = set(categories)
    unknown = categories - set(CROP_TYPES)
    if unknown:
        raise ValueError(f"unsupported categories: {sorted(unknown)} (expected subset of {list(CROP_TYPES)})")

    records = []
    for json_path in split_annotation_paths(split_data, split):
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = Path(data_root) / json_path
        ann = load_annotation(json_path)
        case_id = str(ann.get("case_id") or json_path.parent.name)
        view = str(ann.get("view") or json_path.stem)
        if "full" in categories:
            records.append({
                "case_id": case_id,
                "view": view,
                "crop_type": "full",
                "label": view,
                "sample_key": make_sample_key(case_id, view, "full", view),
                "source_image_path": str(full_image_path(json_path, ann)),
                "box_original": None,
                "annotation_path": str(json_path),
                "partition": split,
            })
        if "tooth" in categories:
            records.extend(_build_tooth_records(ann, json_path, case_id, view, split))
        if "sextant" in categories:
            records.extend(_build_sextant_records(ann, json_path, case_id, view, split))
        if "group" in categories:
            records.extend(_build_group_records(ann, json_path, case_id, view, split))
    return records


def records_hash(records):
    """Stable content hash over the sorted sample_keys (for checkpoint sidecars)."""
    keys = sorted(r["sample_key"] for r in records)
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
    return digest.hexdigest()[:16]


def file_sha256(path, chunk_size=1 << 20):
    """Content hash of the split JSON / manifest files (for sidecars)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def audit_records(records, partition=None):
    """Counts for training/validation records: case, view, crop-type, label stats."""
    stats = {
        "partition": partition,
        "total_records": len(records),
        "cases": len({r["case_id"] for r in records}),
        "views": dict(Counter(r["view"] for r in records)),
        "crop_types": dict(Counter(r["crop_type"] for r in records)),
        "labels_per_crop_type": {},
    }
    by_crop = defaultdict(Counter)
    for r in records:
        by_crop[r["crop_type"]][r["label"]] += 1
    for crop_type, labels in by_crop.items():
        stats["labels_per_crop_type"][crop_type] = {
            "unique_labels": len(labels),
            "total": sum(labels.values()),
        }
    return stats


def print_audit(records, partition=None):
    stats = audit_records(records, partition)
    header = f"records[{stats['partition'] or '?'}]"
    print(f"{header}: total={stats['total_records']} cases={stats['cases']}")
    print(f"  views: {stats['views']}")
    print(f"  crop_types: {stats['crop_types']}")
    for crop_type, info in stats["labels_per_crop_type"].items():
        print(f"  {crop_type}: unique_labels={info['unique_labels']} total={info['total']}")
    return stats


def parse_category_sampling(spec):
    """Parse "full:1.0,tooth:0.5,..." into {crop_type: fraction} (None keeps all)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    result = {}
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            crop_type, fraction = token.split(":", 1)
            result[crop_type.strip()] = float(fraction)
        else:
            result[token] = 1.0
    return result


class DentalSSLData(ExtendedVisionDataset):
    """DINOv2 SSL dataset over case-level split records (see build_records).

    Registered in ``dinov2/data/loaders.py::_parse_dataset_str`` under the name
    ``DentalSSLData``; the dataset string looks like::

        DentalSSLData:root=<data_root>:split_json=<split>:split=TRAIN:categories=full,tooth,sextant,group

    ``__getitem__`` returns ``(PIL image, dummy target)``. The existing
    ``DataAugmentationDINO`` (passed as ``transform=`` by ``make_dataset``)
    performs all global/local multi-crop augmentation; no augmentation is
    repeated inside this dataset.

    The dataset can also export a metadata-only ``<split>_manifest.jsonl``
    (``write_manifest``) and per-crop exposure counts (``exposures``) for
    coverage reporting and cross-baseline leakage checks.
    """

    def __init__(
        self,
        *,
        root: str,
        split_json: str,
        split: str = "TRAIN",
        categories: str = "full,tooth,sextant,group",
        category_sampling: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder_type: DecoderType = DecoderType.ImageDataDecoder,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            image_decoder_type=image_decoder_type,
            **kwargs,
        )
        self.root = root
        self.split_json = str(split_json)
        self.split_name = _SPLIT_ALIASES.get(split, str(split).lower())
        self.categories = [c.strip() for c in str(categories).split(",") if c.strip()]
        self.category_sampling = parse_category_sampling(category_sampling)
        self.seed = kwargs.get("seed", 0)
        self.exposures = Counter()  # sample_key -> number of times fetched

        self.records = build_records(root, self.split_json, self.split_name, self.categories)
        if self.category_sampling and self.split_name == "train":
            self._apply_category_sampling()
        self.image_paths = [r["source_image_path"] for r in self.records]  # compat shim

        logger.info(
            "DentalSSLData: split=%s categories=%s records=%d",
            self.split_name, self.categories, len(self.records),
        )

    # --- category sampling (fixed-ratio, deterministic) -------------------

    def _apply_category_sampling(self):
        """Deterministically downsample per-category pools.

        Keeps the dataset length fixed: for each crop type, keep
        ``ceil(fraction * n_type)`` records selected with a fixed seed, then
        the infinite sampler shuffles the union. A fraction of 1.0 keeps every
        record. DINO's ``SHARDED_INFINITE`` sampler has no per-epoch hook, so
        the subset is fixed for the whole run (unlike MAE/Swin-MAE which
        re-apply the subset each epoch).
        """
        by_crop = {}
        for record in self.records:
            by_crop.setdefault(record["crop_type"], []).append(record)
        kept = []
        for crop_type, records in by_crop.items():
            fraction = self.category_sampling.get(crop_type, 1.0)
            if fraction >= 1.0:
                kept.extend(records)
                continue
            rng = random.Random(self.seed)
            pool = list(records)
            rng.shuffle(pool)
            n_keep = max(1, int(len(pool) * fraction))
            kept.extend(pool[:n_keep])
        self.records = kept
        self.image_paths = [r["source_image_path"] for r in self.records]

    # --- ExtendedVisionDataset interface ----------------------------------

    def get_image_relpath(self, index: int) -> str:
        return self.records[index]["source_image_path"]

    def get_image_data(self, index: int) -> bytes:
        """Bytes of the record's crop (full image, or box re-cut).

        The base class ``__getitem__`` is not used here (we override it to
        return PIL images directly); this method exists for API compatibility.
        """
        record = self.records[index]
        image = PIL.Image.open(record["source_image_path"]).convert("RGB")
        if record["crop_type"] != "full":
            x1, y1, x2, y2 = record["box_original"]
            image = image.crop((int(x1), int(y1), int(x2) + 1, int(y2) + 1))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record = self.records[index]
        if self.split_name == "train":
            self.exposures[record["sample_key"]] += 1
        image = PIL.Image.open(record["source_image_path"]).convert("RGB")
        if record["crop_type"] != "full":
            x1, y1, x2, y2 = record["box_original"]
            image = image.crop((int(x1), int(y1), int(x2) + 1, int(y2) + 1))
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.records)

    # --- manifests / audit ------------------------------------------------

    def write_manifest(self, path: Optional[str] = None) -> Optional[str]:
        """Write a metadata-only ``<split>_manifest.jsonl``."""
        if path is None:
            return None
        return write_manifest(self.records, path)

    def audit(self) -> dict:
        return audit_records(self.records, self.split_name)

    def records_hash(self) -> str:
        return records_hash(self.records)


def build_dataset_str(
    *,
    root: str,
    split_json: str,
    split: str = "TRAIN",
    categories: str = "full,tooth,sextant,group",
    category_sampling: Optional[str] = None,
) -> str:
    """Build the dataset string consumed by ``cfg.train.dataset_path``."""
    parts = [f"DentalSSLData:root={root}:split_json={split_json}:split={split}:categories={categories}"]
    if category_sampling:
        parts.append(f"category_sampling={category_sampling}")
    return ":".join(parts)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI used by ``scripts/dental/train_dental.sh`` to export manifests.

    Builds the train and test partitions, writes ``train_manifest.jsonl`` /
    ``val_manifest.jsonl`` and a ``dental_metadata.json`` sidecar (split_json
    hash, manifest hashes, data root, categories, seed, audits) into
    ``--outdir``, and prints both audits.
    """
    parser = argparse.ArgumentParser(description="DentalSSLData manifest export / audit")
    parser.add_argument("--root", required=True, help="dataset root (data_root)")
    parser.add_argument("--split_json", required=True, help="case-level split JSON")
    parser.add_argument("--outdir", required=True, help="output directory for manifests + sidecar")
    parser.add_argument("--categories", default="full,tooth,sextant,group")
    parser.add_argument("--category_sampling", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "data_root": args.root,
        "split_json": str(args.split_json),
        "split_json_sha256": file_sha256(args.split_json),
        "categories": [c.strip() for c in args.categories.split(",") if c.strip()],
        "category_sampling": args.category_sampling,
        "seed": args.seed,
        "manifests": {},
        "audits": {},
    }

    for split, name in (("train", "train_manifest"), ("test", "val_manifest")):
        dataset = DentalSSLData(
            root=args.root,
            split_json=args.split_json,
            split=split.upper(),
            categories=args.categories,
            category_sampling=args.category_sampling if split == "train" else None,
            seed=args.seed,
        )
        manifest_path = dataset.write_manifest(str(outdir / f"{name}.jsonl"))
        audit = dataset.audit()
        audit["manifest_sha256"] = file_sha256(manifest_path)
        audit["records_hash"] = dataset.records_hash()
        metadata["manifests"][split] = str(manifest_path)
        metadata["audits"][split] = audit
        print_audit(dataset.records, split)

    sidecar = outdir / "dental_metadata.json"
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("wrote manifests + sidecar to %s", outdir)


if __name__ == "__main__":
    main()
