"""Normalize a Roboflow YOLO detection ZIP without changing the source archive."""
from __future__ import annotations

import argparse
import collections
import csv
import io
import math
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
import yaml
from common import dataset_hash, now, sha256_file, write_csv, write_json


def original_id(filename):
    # Roboflow: bola_343_jpg.rf.<hash>.jpg -> bola_343.
    name = filename.split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png)$", "", name, flags=re.I)


def prepare(zip_path, out, groups=None):
    zip_path, out = Path(zip_path).resolve(), Path(out).resolve()
    if out.exists():
        raise RuntimeError(f"Folder sudah ada: {out}. Gunakan --out berbeda agar data lama tetap tersimpan.")
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=out.name + "_tmp_", dir=out.parent))
    try:
        entries = []
        errors, warnings = [], []
        with zipfile.ZipFile(zip_path) as z:
            members = {}
            for inf in z.infolist():
                p = PurePosixPath(inf.filename)
                if p.is_absolute() or ".." in p.parts or "\\" in inf.filename:
                    raise RuntimeError("ZIP memiliki jalur yang tidak valid.")
                if not inf.is_dir():
                    if inf.filename in members and z.read(members[inf.filename]) != z.read(inf):
                        raise RuntimeError(f"Isi ganda berbeda: {inf.filename}")
                    members[inf.filename] = inf
            yamls = [n for n in members if n.endswith("data.yaml")]
            if len(yamls) != 1:
                raise RuntimeError("ZIP harus memiliki satu lokasi data.yaml.")
            yname = yamls[0]
            prefix = str(PurePosixPath(yname).parent)
            prefix = "" if prefix == "." else prefix + "/"
            data = yaml.safe_load(z.read(members[yname]))
            names = data["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            if not names or len(set(names)) != len(names):
                raise RuntimeError("Daftar kelas kosong atau tidak unik.")
            for n in sorted(members):
                relative = n.removeprefix(prefix)
                parts = PurePosixPath(relative).parts
                if len(parts) != 3 or parts[0] not in ("train", "valid", "val", "test") or parts[1] != "images":
                    continue
                if Path(n).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = prefix + f"{parts[0]}/labels/{Path(parts[2]).stem}.txt"
                if label not in members:
                    errors.append(f"Label hilang: {relative}")
                    continue
                entries.append({"image": n, "label": label, "filename": parts[2],
                                "original_id": original_id(parts[2]),
                                "original_split": "val" if parts[0] in ("val", "valid") else parts[0]})
            if not entries:
                raise RuntimeError("Tidak ada gambar pada train/valid/test/images.")
            assigned = {e["image"]: e["original_split"] for e in entries}
            split_method = "existing_roboflow_split"
            if groups:
                with Path(groups).open(encoding="utf-8-sig", newline="") as f:
                    mapping = {r["filename"]: r["session"].strip() for r in csv.DictReader(f)}
                if any(not mapping.get(e["filename"]) for e in entries):
                    raise RuntimeError("Isi kolom session untuk SETIAP filename dalam CSV kelompok/sesi.")
                grouped = collections.defaultdict(list)
                for e in entries:
                    grouped[mapping[e["filename"]]].append(e)
                if len(grouped) < 3:
                    raise RuntimeError("Minimal tiga sesi terpisah untuk train, val, test.")
                ids = sorted(grouped)
                random.Random(42).shuffle(ids)
                targets = {"train": 0.7 * len(entries), "val": 0.2 * len(entries), "test": 0.1 * len(entries)}
                used = collections.Counter()
                for i, gid in enumerate(ids):
                    # Reserve a distinct group for each split, then fill by remaining deficit.
                    split = ("train", "val", "test")[i] if i < 3 else max(targets, key=lambda s: (targets[s] - used[s]) / targets[s])
                    for e in grouped[gid]:
                        assigned[e["image"]] = split
                    used[split] += len(grouped[gid])
                split_method = "session_grouped_seed42"
                warnings.append("Pembagian per sesi mendekati 70/20/10; proporsi tepat bergantung ukuran tiap sesi.")
            else:
                warnings.append("Pembagian asli Roboflow dipertahankan. Tidak ada duplikat persis bukan bukti bebas frame video yang mirip. Untuk klaim generalisasi, pisahkan per video/sesi memakai --groups.")
            counts = {s: collections.Counter(images=0, labels=0, objects=0, empty_labels=0) for s in ("train", "val", "test")}
            dims, exact, pixel_hashes = collections.Counter(), collections.defaultdict(list), collections.defaultdict(list)
            object_sizes, session_rows = [], []
            seen_names = set()
            for e in entries:
                split = assigned[e["image"]]
                if e["filename"] in seen_names:
                    errors.append(f"Nama gambar ganda: {e['filename']}")
                seen_names.add(e["filename"])
                raw, text = z.read(members[e["image"]]), z.read(members[e["label"]]).decode("utf-8-sig")
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img.load()
                    w, h = img.size
                except Exception as exc:
                    errors.append(f"Gambar gagal dibaca {e['filename']}: {exc}")
                    continue
                dims[f"{w}x{h}"] += 1
                exact[__import__("hashlib").sha256(raw).hexdigest()].append(f"{split}/{e['filename']}")
                pixel_hashes[__import__("hashlib").sha256(img.tobytes()).hexdigest()].append(f"{split}/{e['filename']}")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                counts[split]["images"] += 1
                counts[split]["labels"] += 1
                counts[split]["empty_labels"] += not lines
                for line_no, line in enumerate(lines, 1):
                    try:
                        values = list(map(float, line.split()))
                        if len(values) != 5 or not all(math.isfinite(v) for v in values):
                            raise ValueError("harus lima angka finite")
                        cid, cx, cy, bw, bh = values
                        if cid != int(cid) or not 0 <= cid < len(names) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            raise ValueError("ID kelas/koordinat tidak valid")
                        if min(cx - bw/2, cy - bh/2) < -1e-5 or max(cx + bw/2, cy + bh/2) > 1 + 1e-5:
                            raise ValueError("kotak keluar gambar")
                        counts[split]["objects"] += 1
                        object_sizes.append(min(bw*w, bh*h))
                    except ValueError as exc:
                        errors.append(f"{e['filename']} baris {line_no}: {exc}")
                ip = stage / "images" / split / e["filename"]
                lp = stage / "labels" / split / (Path(e["filename"]).stem + ".txt")
                ip.parent.mkdir(parents=True, exist_ok=True)
                lp.parent.mkdir(parents=True, exist_ok=True)
                ip.write_bytes(raw)
                lp.write_text(text, encoding="utf-8")
                session_rows.append({"filename": e["filename"], "original_id": e["original_id"],
                                     "original_split": e["original_split"], "session": ""})
            cross = [v for v in pixel_hashes.values() if len({s.split('/')[0] for s in v}) > 1]
            if cross:
                errors.append(f"Ada {len(cross)} kelompok gambar identik lintas split. Atur pembagian sesi terlebih dahulu.")
            for s, c in counts.items():
                if not c["images"] or not c["objects"]:
                    errors.append(f"Split {s} kosong atau tidak memiliki objek berlabel.")
            (stage / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
            normalized = {"path": out.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": names}
            (stage / "data.yaml").write_text(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True), encoding="utf-8")
            audit = {"created_at": now(), "source_zip": str(zip_path), "source_zip_sha256": sha256_file(zip_path),
                     "names": names, "counts": counts, "dimensions": dict(dims), "split_method": split_method,
                     "duplicate_exact_groups": len([v for v in exact.values() if len(v) > 1]),
                     "cross_split_pixel_duplicates": cross, "errors": errors, "warnings": warnings,
                     "min_object_side_pixels": min(object_sizes) if object_sizes else None,
                     "dataset_sha256": dataset_hash(stage)}
            write_json(stage / "audit_dataset.json", audit)
            write_csv(stage / "peta_sesi.csv", session_rows)
            (stage / "catatan_dataset.txt").write_text("\n".join(warnings + errors) + "\n", encoding="utf-8")
        stage.replace(out)
        print(f"Dataset tersimpan di {out}")
        for s, c in counts.items():
            print(f"{s}: {c['images']} gambar, {c['objects']} objek, {c['empty_labels']} label kosong")
        print("Kesalahan:", len(errors), "| Sidik jari:", audit["dataset_sha256"][:16])
        for warning in warnings:
            print("CATATAN:", warning)
        if errors:
            raise RuntimeError("Lihat audit_dataset.json. Training diblokir sampai dataset diperbaiki.")
        return audit
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("dataset_v1"))
    p.add_argument("--groups", type=Path, help="CSV hasil peta_sesi.csv dengan session terisi per video/sesi asli")
    a = p.parse_args()
    prepare(a.zip, a.out, a.groups)


if __name__ == "__main__":
    main()
