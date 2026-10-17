"""Shared validation and reporting helpers. Importing this file does not load YOLO."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path

import yaml


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_hash(root):
    """Hash paths, split membership, names, and bytes of every image/label."""
    root = Path(root)
    h = hashlib.sha256()
    files = [root / "classes.txt"]
    for folder in ("images", "labels"):
        files += sorted(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix != ".cache")
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def read_data(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = path.parent / root
    return path, root.resolve(), data


def checked_data(path):
    path, root, data = read_data(path)
    audit = read_json(root / "audit_dataset.json")
    if audit["errors"]:
        raise RuntimeError("Dataset memiliki kesalahan. Perbaiki berdasarkan audit_dataset.json.")
    current = dataset_hash(root)
    if current != audit["dataset_sha256"]:
        raise RuntimeError("Gambar/label berubah setelah audit. Siapkan ulang ke folder baru.")
    for split in ("train", "val", "test"):
        expected = f"images/{'val' if split == 'val' else split}"
        if data.get(split) != expected:
            raise RuntimeError(f"Jalur {split} harus {expected}; gunakan YAML hasil program 01.")
    if data.get("names") != audit["names"]:
        raise RuntimeError("Daftar kelas YAML berubah setelah audit.")
    return path, root, audit, current


def require_gpu(device=0):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Jalankan 00_cek_gpu.py; jangan training dengan CPU tanpa sengaja.")
    if not 0 <= device < torch.cuda.device_count():
        raise RuntimeError(f"GPU {device} tidak tersedia.")
    prop = torch.cuda.get_device_properties(device)
    return {
        "device": device, "gpu": prop.name,
        "vram_gb": round(prop.total_memory / 1024**3, 2),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(), "os": platform.platform(),
    }


def metric_dict(metrics, names):
    out = {
        "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "val_batch_inference_ms": float(metrics.speed.get("inference", 0)),
        "class_mAP50_95": {str(n): float(v) for n, v in zip(names, metrics.box.maps)},
    }
    if not all(math.isfinite(out[k]) for k in ("mAP50", "mAP50_95", "precision", "recall")):
        raise RuntimeError("Metrik bukan angka finite; periksa hasil evaluasi.")
    return out


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def read_training_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(f)]


def choose_best(results):
    # Ties retain E1 as the simpler baseline. Test is never used for selection.
    return max(results, key=lambda name: results[name]["val"]["mAP50_95"])


def comparison_rows(manifest):
    rows = []
    for name in ("E1_pretrained", "E3_tuning"):
        exp = manifest["experiments"][name]
        v = exp["val"]
        b = exp.get("benchmark", {})
        rows.append({
            "eksperimen": name, "imgsz": exp["imgsz"], "batch": exp["actual_batch"],
            "mAP50_val": v["mAP50"], "mAP50_95_val": v["mAP50_95"],
            "precision_val": v["precision"], "recall_val": v["recall"],
            "waktu_train_menit": round(exp["elapsed_train_seconds"] / 60, 3),
            "detector_batch1_mean_ms": b.get("mean_ms", ""),
            "detector_batch1_p95_ms": b.get("p95_ms", ""),
            "detector_batch1_fps": b.get("detector_fps", ""),
            "bobot": exp["best_pt"],
        })
    return rows
