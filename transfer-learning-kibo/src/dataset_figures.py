"""Gambar untuk datasheet: distribusi kelas per split dan contoh crop tiap kelas."""
import argparse, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/cls"); ap.add_argument("--out", default="docs/img")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
classes = sorted(os.listdir(f"{a.data}/train")); splits = ["train", "valid", "test"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 4)); w = 0.27; col = ["#4a6fa5", "#e9a23b", "#2a9d8f"]
for i, s in enumerate(splits):
    n = [len(glob.glob(f"{a.data}/{s}/{c}/*.png")) for c in classes]
    ax.bar(np.arange(len(classes)) + (i - 1) * w, n, w, label=f"{s} (n={sum(n)})", color=col[i])
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=35, ha="right"); ax.set_ylabel("jumlah crop")
ax.set_title("Distribusi kelas per split", loc="left", fontweight="bold"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{a.out}/distribusi_kelas.png", dpi=140)
fig, axs = plt.subplots(len(classes), 8, figsize=(10, 1.25 * len(classes)))
rng = np.random.RandomState(0)
for r, c in enumerate(classes):
    fs = sorted(glob.glob(f"{a.data}/train/{c}/*.png")); pick = rng.choice(len(fs), 8, replace=False)
    for j, k in enumerate(pick):
        axs[r, j].imshow(Image.open(fs[k]).convert("RGB")); axs[r, j].axis("off")
    axs[r, 0].text(-0.1, 0.5, c, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=9)
fig.suptitle("Contoh crop (train), 8 acak per kelas", fontweight="bold"); fig.tight_layout(); fig.savefig(f"{a.out}/contoh_crop.png", dpi=120)
