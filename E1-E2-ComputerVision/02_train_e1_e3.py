"""Train E1 and E3 on one frozen dataset; never evaluate the test split here."""
from __future__ import annotations

import argparse
import gc
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from common import (checked_data, choose_best, comparison_rows, metric_dict, now,
                    read_json, read_training_csv, require_gpu, sha256_file, write_csv, write_json)


def benchmark(model, dataset_root, imgsz, device, count=24):
    """Single-image detector latency, including Python/pre/postprocess, not camera/ROS."""
    import numpy as np
    import torch
    from PIL import Image
    paths = sorted(p for p in (dataset_root / "images/val").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    selected = random.Random(42).sample(paths, min(count, len(paths)))
    images = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy() for p in selected]
    cfg = dict(imgsz=imgsz, device=device, half=False, conf=0.25, verbose=False, save=False)
    for _ in range(3):
        model.predict(source=images[0], **cfg)
    times = []
    for _ in range(2):
        for image in images:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            model.predict(source=image, **cfg)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "detector_fps": float(1000 / np.mean(times)),
            "images": len(images), "passes": 2, "warmup": 3, "batch": 1, "precision": "FP32",
            "scope": "Preloaded BGR image -> prediction; includes Python/preprocess/inference/postprocess, excludes file IO, camera, ROS and display.",
            "filenames": [p.name for p in selected]}


def train_one(name, size, manifest, run, manifest_path):
    import torch
    from ultralytics import YOLO
    config = manifest["config"]
    exp = manifest["experiments"].setdefault(name, {"imgsz": size, "elapsed_train_seconds": 0})
    if exp.get("status") == "completed":
        if sha256_file(exp["best_pt"]) != exp["best_sha256"]:
            raise RuntimeError(f"Bobot {name} berubah setelah evaluasi.")
        print(f"{name} selesai sebelumnya; memakai hasil tersimpan.")
        return
    destination = run / name
    best, last = destination / "weights/best.pt", destination / "weights/last.pt"
    csv_path = destination / "results.csv"
    train_complete = exp.get("training_complete", False)
    if best.exists() and csv_path.exists() and len(read_training_csv(csv_path)) >= config["epochs"]:
        train_complete = True
    if not train_complete:
        # A NEW object and the SAME initial pretrained artifact for each experiment.
        resuming = last.exists()
        model = YOLO(str(last if resuming else manifest["initial_weights"]))
        cfg = dict(data=manifest["data_yaml"], epochs=config["epochs"], imgsz=size,
                   batch=config["batch"], device=config["device"], workers=config["workers"],
                   seed=42, deterministic=True, amp=True, cache=False, plots=True,
                   optimizer="AdamW", lr0=0.001, patience=config["epochs"],
                   project=str(run), name=name, exist_ok=True, pretrained=True,
                   save=True, val=True, close_mosaic=10, verbose=True)
        exp["status"] = "training"
        write_json(manifest_path, manifest)
        start = time.perf_counter()
        try:
            if resuming:
                print(f"Melanjutkan {name} dari {last}")
                model.train(resume=True, data=manifest["data_yaml"], device=config["device"], workers=config["workers"])
            else:
                model.train(**cfg)
            exp["actual_batch"] = int(getattr(model.trainer, "batch_size", model.trainer.args.batch))
            if exp["actual_batch"] != config["batch"]:
                raise RuntimeError("Batch berubah otomatis setelah kehabisan VRAM. Buat run BARU dengan batch lebih kecil untuk KEDUA eksperimen.")
            exp["training_complete"] = True
        except BaseException as exc:
            exp["status"] = "interrupted"
            exp["last_error"] = str(exc)
            raise
        finally:
            exp["elapsed_train_seconds"] += time.perf_counter() - start
            write_json(manifest_path, manifest)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        exp.setdefault("actual_batch", config["batch"])
        exp["training_complete"] = True
    if not best.exists():
        raise RuntimeError(f"Tidak menemukan {best}")
    # Verify args.yaml even when resuming a finished training.
    import yaml
    actual_args = yaml.safe_load((destination / "args.yaml").read_text(encoding="utf-8"))
    if int(actual_args["batch"]) != config["batch"] or int(actual_args["imgsz"]) != size:
        raise RuntimeError("Konfigurasi checkpoint berbeda dari rancangan perbandingan.")
    exp["status"] = "validation"
    write_json(manifest_path, manifest)
    model = YOLO(str(best))
    metrics = model.val(data=manifest["data_yaml"], split="val", imgsz=size, batch=config["batch"],
                        device=config["device"], workers=config["workers"], half=False,
                        plots=True, project=str(run / "evaluasi_val"), name=name, exist_ok=True, verbose=False)
    exp["val"] = metric_dict(metrics, manifest["names"])
    exp["benchmark"] = benchmark(model, Path(manifest["dataset_root"]), size, config["device"])
    exp["best_pt"], exp["best_sha256"] = str(best), sha256_file(best)
    exp["status"], exp["finished_at"] = "completed", now()
    write_json(manifest_path, manifest)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_training(args):
    import ultralytics
    from ultralytics import YOLO
    if args.resume:
        run = args.resume.resolve()
        manifest_path = run / "manifest.json"
        manifest = read_json(manifest_path)
        data, root, audit, fingerprint = checked_data(manifest["data_yaml"])
        if fingerprint != manifest["dataset_sha256"]:
            raise RuntimeError("Dataset berubah; resume tidak boleh memakai data yang berbeda.")
        if sha256_file(manifest["initial_weights"]) != manifest["initial_weights_sha256"]:
            raise RuntimeError("Bobot awal berubah setelah run dimulai.")
        current_env = require_gpu(manifest["config"]["device"])
        if (current_env["torch"] != manifest["environment"]["torch"]
                or ultralytics.__version__ != manifest["environment"]["ultralytics"]
                or current_env["gpu"] != manifest["environment"]["gpu"]):
            raise RuntimeError("GPU atau versi torch/ultralytics berubah. Gunakan environment yang sama untuk resume.")
    else:
        if not args.nim or not args.nama:
            raise RuntimeError("Isi --nim dan --nama.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.nim):
            raise RuntimeError("NIM hanya boleh berisi huruf, angka, _ atau -.")
        if args.batch < 1 or args.epochs < 1:
            raise RuntimeError("Batch dan epochs harus positif.")
        data, root, audit, fingerprint = checked_data(args.data)
        environment = require_gpu(args.device)
        environment["ultralytics"] = ultralytics.__version__
        weights = args.model.resolve()
        if weights.suffix != ".pt":
            raise RuntimeError("E1/E3 membutuhkan bobot pretrained .pt.")
        # Auto-download a recognized Ultralytics .pt name if it is not present.
        starter = YOLO(str(weights))
        if not weights.exists():
            weights = Path(starter.ckpt_path).resolve()
        del starter
        if not weights.exists():
            raise RuntimeError("Berkas pretrained tidak ditemukan setelah pemuatan.")
        name = args.run_name or f"{args.nim}_{time.strftime('%Y%m%d_%H%M%S')}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise RuntimeError("run-name hanya boleh berisi huruf, angka, _ atau -.")
        run = args.out.resolve() / name
        run.mkdir(parents=True, exist_ok=False)
        manifest_path = run / "manifest.json"
        manifest = {"created_at": now(), "nim": args.nim, "nama": args.nama,
                    "data_yaml": str(data), "dataset_root": str(root), "dataset_sha256": fingerprint,
                    "names": audit["names"], "dataset_audit": audit, "environment": environment,
                    "initial_weights": str(weights), "initial_weights_sha256": sha256_file(weights),
                    "config": {"epochs": args.epochs, "batch": args.batch, "device": args.device, "workers": args.workers,
                               "seed": 42, "optimizer": "AdamW", "lr0": 0.001, "amp": True},
                    "experiments": {}, "selection_rule": "highest validation mAP50-95; E1 wins ties"}
        write_json(manifest_path, manifest)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run / "requirements_used.txt").write_text(freeze.stdout, encoding="utf-8")
    for name, size in (("E1_pretrained", 416), ("E3_tuning", 640)):
        train_one(name, size, manifest, run, manifest_path)
    manifest["selected"] = choose_best(manifest["experiments"])
    manifest["training_finished_at"] = now()
    write_json(manifest_path, manifest)
    write_csv(run / f"hasil_praktik_{manifest['nim']}.csv", comparison_rows(manifest))
    print(f"\nSELESAI TRAIN/VAL: {run}")
    print("Model terpilih berdasarkan VAL:", manifest["selected"])
    print("Test belum dijalankan. Lanjut: python 03_uji_dan_analisis.py --run", f'"{run}"')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("dataset_v1/data.yaml"))
    p.add_argument("--nim")
    p.add_argument("--nama")
    p.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="0 adalah titik awal yang mudah dipakai di Windows")
    p.add_argument("--out", type=Path, default=Path("hasil_praktik"))
    p.add_argument("--run-name")
    p.add_argument("--resume", type=Path, help="Folder run yang terhenti; konfigurasi aslinya dipertahankan")
    run_training(p.parse_args())


if __name__ == "__main__":
    # Required on Windows when PyTorch uses multiprocessing.
    from multiprocessing import freeze_support
    freeze_support()
    main()
