"""Inspect the selected model on validation images, a new video, or a webcam."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import read_json, require_gpu, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--source", help="Default: gambar validasi. Bisa path gambar/video atau 0 untuk webcam.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    manifest = read_json(a.run / "manifest.json")
    exp = manifest["experiments"][manifest["selected"]]
    if sha256_file(exp["best_pt"]) != exp["best_sha256"]:
        raise RuntimeError("Bobot model berubah setelah validasi.")
    require_gpu(manifest["config"]["device"])
    from ultralytics import YOLO
    source = a.source or str(Path(manifest["dataset_root"]) / "images/val")
    if source.isdecimal():
        source = int(source)
    for _ in YOLO(exp["best_pt"]).predict(source=source, imgsz=exp["imgsz"], conf=a.conf,
                device=manifest["config"]["device"], half=False, stream=True,
                show=a.show, save=True, save_txt=True, save_conf=True,
                project=str(a.run.resolve() / "prediksi"), name="contoh", exist_ok=False):
        pass


if __name__ == "__main__":
    main()
