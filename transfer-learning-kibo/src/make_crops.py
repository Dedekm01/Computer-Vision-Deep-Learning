"""Ubah dataset deteksi objek (YOLOv8) menjadi dataset klasifikasi dengan meng-crop tiap bounding box.

Contoh:
    python src/make_crops.py --src data/yolov8 --out data/cls
Struktur keluaran: <out>/{train,valid,test}/<nama_kelas>/<gambar>_<idx>.png
Split train/valid/test mengikuti split Roboflow (per gambar sumber, jadi crop dari satu gambar
tidak pernah tersebar ke dua split).
"""
import argparse, collections, glob, os
import numpy as np, yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/yolov8", help="folder hasil ekspor YOLOv8 (berisi data.yaml, train/, valid/, test/)")
ap.add_argument("--out", default="data/cls")
ap.add_argument("--margin", type=float, default=0.10, help="margin tambahan di sekeliling box (fraksi)")
ap.add_argument("--min-size", type=int, default=8, help="crop dengan sisi < nilai ini (px) dibuang")
ap.add_argument("--skip-classes", nargs="*", default=["aexample"], help="kelas yang dibuang (placeholder)")
a = ap.parse_args()

names = yaml.safe_load(open(os.path.join(a.src, "data.yaml")))["names"]
cnt, sizes = collections.Counter(), []
for split in ["train", "valid", "test"]:
    for lf in sorted(glob.glob(f"{a.src}/{split}/labels/*.txt")):
        base = os.path.splitext(os.path.basename(lf))[0]
        im = Image.open(glob.glob(f"{a.src}/{split}/images/{base}.*")[0]).convert("RGB")
        W, H = im.size
        for k, line in enumerate(open(lf)):
            p = line.split()
            if len(p) < 5:
                continue
            cls = names[int(p[0])]
            if cls in a.skip_classes:
                continue
            x, y, w, h = map(float, p[1:5])
            x0 = max(0, int((x - w / 2 * (1 + a.margin)) * W)); x1 = min(W, int((x + w / 2 * (1 + a.margin)) * W))
            y0 = max(0, int((y - h / 2 * (1 + a.margin)) * H)); y1 = min(H, int((y + h / 2 * (1 + a.margin)) * H))
            if x1 - x0 < a.min_size or y1 - y0 < a.min_size:
                continue
            d = f"{a.out}/{split}/{cls}"
            os.makedirs(d, exist_ok=True)
            im.crop((x0, y0, x1, y1)).save(f"{d}/{base}_{k}.png")
            cnt[(split, cls)] += 1; sizes.append((x1 - x0, y1 - y0))

classes = [n for n in names if n not in a.skip_classes]
print("crop per split:", {s: sum(v for (sp, _), v in cnt.items() if sp == s) for s in ["train", "valid", "test"]})
print("ukuran crop rata-rata (w,h):", np.mean(sizes, 0).round(1), "min:", np.min(sizes, 0), "max:", np.max(sizes, 0))
print(f"{'kelas':14}{'train':>7}{'valid':>7}{'test':>7}")
for c in classes:
    print(f"{c:14}" + "".join(f"{cnt[(s, c)]:7d}" for s in ["train", "valid", "test"]))
