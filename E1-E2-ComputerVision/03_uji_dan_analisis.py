"""Evaluate only the validation-selected model on test, then write real reports."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import (checked_data, choose_best, comparison_rows, metric_dict, now,
                    read_json, read_training_csv, require_gpu, sha256_file, write_csv, write_json)


def plot_results(run, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = ["E1_pretrained", "E3_tuning"]
    labels = ["E1: input 416", "E3: input 640"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(4)
    keys = ["mAP50", "mAP50_95", "precision", "recall"]
    for i, name in enumerate(names):
        v = manifest["experiments"][name]["val"]
        axes[0].bar(x + (i - .5) * .35, [v[k] for k in keys], width=.35, label=labels[i])
    axes[0].set_xticks(x, ["mAP50", "mAP50-95", "Precision", "Recall"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Perbandingan pada data validasi")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=.2)
    times = [manifest["experiments"][n]["benchmark"]["mean_ms"] for n in names]
    axes[1].bar(labels, times, color=["#277DA1", "#F9844A"])
    axes[1].set_ylabel("Waktu rata-rata detektor, ms/gambar")
    axes[1].set_title("Batch 1, FP32; tanpa kamera/ROS")
    for i, t in enumerate(times):
        axes[1].text(i, t, f"{t:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(run / "perbandingan_E1_E3.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    series = [("metrics/mAP50-95(B)", "mAP50-95 validasi"), ("metrics/recall(B)", "Recall validasi"),
              ("train/box_loss", "Box loss train"), ("val/box_loss", "Box loss validasi")]
    for name, label in zip(names, labels):
        rows = read_training_csv(run / name / "results.csv")
        for ax, (key, title) in zip(axes.flat, series):
            points = [(r.get("epoch", i + 1), r[key]) for i, r in enumerate(rows) if key in r]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], label=label)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.2)
    for ax in axes.flat:
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(run / "kurva_E1_E3.png", dpi=180)
    plt.close(fig)


def write_report(run, manifest, test):
    rows = comparison_rows(manifest)
    selected = manifest["selected"]
    e1, e3 = [manifest["experiments"][n] for n in ("E1_pretrained", "E3_tuning")]
    delta = e3["val"]["mAP50_95"] - e1["val"]["mAP50_95"]
    change = "meningkat" if delta > 0 else "menurun" if delta < 0 else "tetap"
    dt = e3["benchmark"]["mean_ms"] - e1["benchmark"]["mean_ms"]
    speed_text = f"{abs(dt):.2f} ms lebih {'lambat' if dt > 0 else 'cepat'}" if dt else "sama"
    counts = manifest["dataset_audit"]["counts"]
    text = [
        "# Analisis E1 dan E3: deteksi bola humanoid soccer", "",
        f"Nama: {manifest['nama']}  ", f"NIM: {manifest['nim']}  ",
        f"Dibuat: {now()}  ", f"GPU: {manifest['environment']['gpu']}  ",
        f"Dataset SHA256: `{manifest['dataset_sha256']}`", "",
        "## Tujuan dan rancangan", "",
        "Percobaan menguji pengaruh perubahan ukuran input 416 menjadi 640 pada deteksi bola. "
        "E1 dan E3 dimulai secara terpisah dari berkas pretrained yang sama, dengan dataset dan pembagian yang sama. "
        "E2 tidak dijalankan. Percobaan ini tidak mengukur keuntungan pretrained dibanding training dari bobot acak.", "",
        f"Epoch: {manifest['config']['epochs']}; batch: {manifest['config']['batch']}; seed: 42; "
        "optimizer: AdamW; lr0: 0.001; training AMP aktif. Model dipilih dari mAP50-95 validasi, bukan test.", "",
        "## Dataset", "",
        "| Bagian | Gambar | Kotak objek | Label kosong |", "|---|---:|---:|---:|",
    ]
    for split, c in counts.items():
        text.append(f"| {split} | {c['images']} | {c['objects']} | {c['empty_labels']} |")
    text += ["", "Kelas: " + ", ".join(manifest["names"]) + ".", "",
             "Metode pembagian: " + manifest["dataset_audit"]["split_method"] + ".", ""]
    for warning in manifest["dataset_audit"]["warnings"]:
        text += ["Catatan: " + warning, ""]
    text += ["## Hasil validasi", "",
             "| Eksperimen | Input | mAP50 | mAP50-95 | Precision | Recall | Train (menit) | Detektor batch 1 (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['eksperimen']} | {r['imgsz']} | {r['mAP50_val']:.4f} | {r['mAP50_95_val']:.4f} | "
                    f"{r['precision_val']:.4f} | {r['recall_val']:.4f} | {r['waktu_train_menit']:.2f} | {r['detector_batch1_mean_ms']:.2f} |")
    text += ["", "## Pembahasan", "",
             f"Dari E1 ke E3, mAP50-95 validasi {change} sebesar {abs(delta)*100:.2f} poin persentase "
             f"(selisih skor {delta:+.4f}). Waktu detektor E3 {speed_text} daripada E1 pada prosedur benchmark yang sama. "
             "Satu kali percobaan per konfigurasi belum mengukur variasi antar-seed.", "",
             f"Model terpilih adalah **{selected}**, karena mempunyai mAP50-95 validasi tertinggi "
             "(jika sama, baseline E1 dipertahankan). E3 yang lebih rendah tetap merupakan hasil eksperimen yang sah.", "",
             "Precision menunjukkan proporsi prediksi yang benar, sedangkan recall menunjukkan proporsi objek yang ditemukan. "
             "Untuk soccer, periksa juga contoh bola jauh, tertutup sebagian, dan blur di hasil prediksi validasi. "
             "Peningkatan metrik keseluruhan saja belum membuktikan perbaikan khusus pada bola jauh.", "",
             "Benchmark memakai batch 1, FP32, 24 gambar validasi atau sebanyak yang tersedia, dua putaran, "
             "dan tiga pemanasan. Gambar telah dimuat sebelum pengukuran. Waktu mencakup pemanggilan Python, "
             "preprocessing, inference, dan postprocessing. Kamera, baca file, ROS, display, dan kontrol robot tidak dihitung. "
             "Nilai FPS detektor bukan FPS keseluruhan robot. GPU lain perlu diukur lagi.", "",
             "## Kurva pelatihan", ""]
    for name in ("E1_pretrained", "E3_tuning"):
        history = read_training_csv(run / name / "results.csv")
        key = "metrics/mAP50-95(B)"
        if history and all(key in r for r in history):
            peak = max(history, key=lambda r: r[key])
            tail = sum(r[key] for r in history[-min(5, len(history)):]) / min(5, len(history))
            text += [f"{name}: nilai mAP50-95 tercatat tertinggi {peak[key]:.4f} pada epoch {int(peak.get('epoch', 0))}; "
                     f"rata-rata hingga lima epoch terakhir {tail:.4f}. Lihat kurva train/val loss untuk menilai polanya. "
                     "Selisih ini saja tidak cukup untuk memastikan overfitting.", ""]
    t = test["metrics"]
    text += ["## Pengujian akhir", "",
             f"Hanya model {selected} yang dievaluasi pada split test. Hasil: mAP50 **{t['mAP50']:.4f}**, "
             f"mAP50-95 **{t['mAP50_95']:.4f}**, precision **{t['precision']:.4f}**, recall **{t['recall']:.4f}**.", "",
             "Perbedaan skor val dan test dapat dipengaruhi variasi sampel. Jika frame berdekatan masih tersebar antarsplit, "
             "hasil tidak boleh diklaim sebagai bukti generalisasi ke sesi pertandingan baru. "
             "Sesudah hasil test dibuka, jangan memilih ulang E1/E3 berdasarkan skor test tersebut.", "",
             "## Kesimpulan awal dari angka", "",
             f"Untuk dataset dan konfigurasi ini, {selected} terpilih berdasarkan validasi. "
             f"Perubahan input 416 ke 640 membuat skor mAP50-95 validasi {change}. "
             "Kelayakan pada robot tetap perlu ditentukan dari recall yang dibutuhkan serta pengukuran latensi di perangkat robot.", "",
             "## Pemeriksaan manual untuk laporan kelompok", "",
             "Lengkapi uraian misi robot, kamera dan posisinya, target FPS/latensi/recall yang ditetapkan kelompok, "
             "asal serta sesi rekaman, dan contoh keberhasilan/kegagalan deteksi yang benar-benar diamati. "
             "Target dan pengamatan tersebut tidak dibuat otomatis oleh program.", "",
             "Bukti: E1_pretrained/results.png, E3_tuning/results.png, confusion matrix/kurva pada evaluasi_val "
             "dan evaluasi_test, CSV per epoch, manifest.json, serta requirements_used.txt.", "",
             "Grafik tambahan: perbandingan_E1_E3.png dan kurva_E1_E3.png.", ""]
    report = "\n".join(text)
    (run / "analisis_otomatis.md").write_text(report, encoding="utf-8")
    # Keep the user's later observations in Read.Me when regenerating charts.
    if not (run / "Read.Me.md").exists():
        (run / "Read.Me.md").write_text(report, encoding="utf-8")
    (run / f"hasil_praktik_{manifest['nim']}.txt").write_text(report, encoding="utf-8")
    write_csv(run / f"hasil_praktik_{manifest['nim']}.csv", rows)
    write_csv(run / f"hasil_test_{manifest['nim']}.csv", [{"model": selected, **{k: v for k, v in t.items() if not isinstance(v, dict)}}])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--retry-failed-test", action="store_true", help="Ulangi hanya jika evaluasi final sebelumnya gagal secara teknis; percobaan dicatat")
    a = p.parse_args()
    run = a.run.resolve()
    manifest = read_json(run / "manifest.json")
    _, _, _, fingerprint = checked_data(manifest["data_yaml"])
    if fingerprint != manifest["dataset_sha256"]:
        raise RuntimeError("Dataset berubah sesudah training.")
    exps = manifest["experiments"]
    if set(exps) != {"E1_pretrained", "E3_tuning"} or any(e.get("status") != "completed" for e in exps.values()):
        raise RuntimeError("Kedua eksperimen harus selesai terlebih dahulu.")
    if manifest.get("selected") != choose_best(exps):
        raise RuntimeError("Pemilihan model tidak sesuai aturan mAP50-95 validasi.")
    selected = manifest["selected"]
    exp = exps[selected]
    weight_hash = sha256_file(exp["best_pt"])
    if weight_hash != exp["best_sha256"]:
        raise RuntimeError("Bobot model berubah sesudah validasi.")
    test_path = run / "hasil_test.json"
    if test_path.exists():
        test = read_json(test_path)
        if (test["selected"], test["weight_sha256"], test["dataset_sha256"]) != (selected, weight_hash, fingerprint):
            raise RuntimeError("Identitas hasil test berbeda; jangan memilih ulang setelah membuka test.")
        print("Memakai hasil test tersimpan; test tidak dievaluasi ulang.")
    else:
        marker_path = run / "status_test.json"
        marker = read_json(marker_path) if marker_path.exists() else {"attempts": []}
        if marker["attempts"] and not a.retry_failed_test:
            raise RuntimeError("Evaluasi test sebelumnya terhenti. Jika kegagalan teknis sudah diperbaiki, gunakan --retry-failed-test; bobot dan data harus tetap sama.")
        if marker.get("weight_sha256", weight_hash) != weight_hash or marker.get("selected", selected) != selected:
            raise RuntimeError("Bobot/model test berubah setelah percobaan pertama.")
        require_gpu(manifest["config"]["device"])
        from ultralytics import YOLO
        marker.update(selected=selected, weight_sha256=weight_hash, dataset_sha256=fingerprint)
        marker["attempts"].append({"started_at": now(), "status": "running"})
        write_json(marker_path, marker)
        try:
            metrics = YOLO(exp["best_pt"]).val(
                data=manifest["data_yaml"], split="test", imgsz=exp["imgsz"], batch=manifest["config"]["batch"],
                device=manifest["config"]["device"], workers=manifest["config"]["workers"], half=False,
                plots=True, project=str(run / "evaluasi_test"), name=selected, exist_ok=True, verbose=False)
            test = {"evaluated_at": now(), "selected": selected, "weight_sha256": weight_hash,
                    "dataset_sha256": fingerprint, "metrics": metric_dict(metrics, manifest["names"])}
            write_json(test_path, test)
            marker["attempts"][-1].update(status="completed", finished_at=now())
        except BaseException as exc:
            marker["attempts"][-1].update(status="failed", error=str(exc), finished_at=now())
            raise
        finally:
            write_json(marker_path, marker)
    write_report(run, manifest, test)
    plot_results(run, manifest)
    print(f"\nLaporan: {run / 'Read.Me.md'}")
    print(f"Model terpilih: {selected}; test mAP50-95: {test['metrics']['mAP50_95']:.4f}")
    print("Lengkapi pengamatan contoh deteksi dan konteks robot pada laporan.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
