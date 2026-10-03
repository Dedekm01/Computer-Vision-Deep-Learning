# Transfer Learning pada Dataset Kibo-RPC: MobileNetV3-Large vs ResNet-18 vs ResNet-50

Praktikum **RET503 – Pertemuan 3 (P2)**: membandingkan tiga model pretrained ImageNet untuk mengklasifikasikan
11 jenis objek "harta karun" dari dataset lomba Kibo-RPC, lalu memilih model yang paling cocok untuk robot.

![Perbandingan model](results/figures/perbandingan_model.png)

## Ringkasan hasil

Rata-rata 3 kali pelatihan (seed 1, 2, 3) pada test set 175 gambar:

| Model | Akurasi test | Macro F1 | Loss test | Latensi CPU* | Parameter |
|---|---|---|---|---|---|
| MobileNetV3-Large | 99,81% ± 0,27 | 99,80% | 0,073 | **11,2 ms** | 4,2 jt |
| ResNet-18 | **100,00% ± 0,00** | 100,00% | 0,054 | 20,3 ms | 11,2 jt |
| ResNet-50 | 99,62% ± 0,27 | 99,59% | **0,044** | 46,5 ms | 23,5 jt |

\* CPU server 1 thread, batch 1, input 128×128 px. Angka absolutnya akan berbeda di Raspberry Pi/Jetson; urutan antar model yang bisa dipakai.

**Kesimpulan singkat**
- Akurasi ketiga model **praktis sama**. Selisih terbesar hanya 1 gambar dari 175 (0,57%) dan itu masih dalam batas noise antar-seed, jadi tidak ada pemenang akurasi yang pasti.
- **ResNet-18** paling konsisten (100% di ketiga seed). **MobileNetV3-Large** hampir sama akuratnya tetapi ±1,8× lebih cepat dan ±2,7× lebih kecil dari ResNet-18, sehingga paling cocok untuk robot/edge. **ResNet-50** paling lambat (±4× MobileNet) tanpa keuntungan akurasi.
- Dataset ini mudah (objek berupa ikon garis hitam yang jelas berbeda), sehingga hasilnya jenuh. Detail dan keterbatasan ada di [`docs/HASIL.md`](docs/HASIL.md).

## Isi repo

```
.
├── README.md              <- halaman ini
├── DATASHEET.md           <- datasheet dataset (asal, isi, split, keterbatasan)
├── docs/HASIL.md          <- penjelasan tiap chart, tabel per-seed, analisis
├── docs/img/              <- distribusi kelas & contoh crop
├── src/
│   ├── make_crops.py      <- YOLOv8 (deteksi) -> dataset klasifikasi (crop bounding box)
│   ├── train.py           <- transfer learning 3 model
│   ├── charts.py          <- chart perbandingan, kurva loss, confusion matrix, CSV ringkasan
│   └── dataset_figures.py <- gambar untuk datasheet
├── results/
│   ├── figures/           <- chart hasil
│   ├── raw/               <- JSON hasil mentah tiap model & seed (histori epoch, confusion matrix)
│   ├── ringkasan_model.csv
│   └── hasil_per_seed.csv
├── data/data.yaml.contoh  <- konfigurasi kelas dari ekspor Roboflow
├── run_all.sh             <- jalankan seluruh pipeline
└── requirements.txt
```

## Cara menjalankan

1. Unduh dataset dari Roboflow: <https://universe.roboflow.com/brail-nyx-88/crop-aja> (versi 10, format **YOLOv8**) lalu ekstrak ke `data/yolov8/` (harus berisi `data.yaml`, `train/`, `valid/`, `test/`). Gambar tidak disertakan di repo.
2. Pasang dependensi dan jalankan:

```bash
pip install -r requirements.txt
./run_all.sh            # crop -> train 3 seed -> chart (±35 menit di CPU 2 core)
```

Atau langkah demi langkah:

```bash
python src/make_crops.py --src data/yolov8 --out data/cls
python src/train.py --seed 1 --epochs 8                 # semua model; --models ResNet-18 untuk satu model
python src/charts.py                                    # butuh JSON di results/raw
```

Tidak butuh GPU. Bobot ImageNet diunduh otomatis oleh `pytorchcv` saat pertama kali dijalankan.

## Metodologi

**1. Data.** Dataset asli adalah dataset *deteksi objek* (799 gambar 320×220, 12 kelas, beberapa objek per gambar). Tugas praktikum adalah *klasifikasi*, jadi tiap bounding box di-crop (margin 10%) menjadi satu gambar klasifikasi. Hasilnya **1.903 crop**: 1.558 train / 170 valid / 175 test, 11 kelas. Kelas `aexample` dibuang karena hanya 1 contoh (placeholder). Split mengikuti split Roboflow, yang dibagi per gambar sumber, sehingga crop dari satu gambar tidak tersebar ke dua split. Detail di [`DATASHEET.md`](DATASHEET.md).

**2. Model.** Tiga arsitektur dengan bobot pretrained ImageNet: MobileNetV3-Large, ResNet-18, ResNet-50. Classifier 1000 kelas diganti dengan 11 kelas. Pooling akhir diganti `AdaptiveAvgPool2d(1)` karena pooling asli berukuran tetap 7×7 (untuk input 224), sedangkan di sini input 128.

**3. Strategi transfer learning: partial fine-tuning.** Seluruh lapisan dibekukan kecuali **stage terakhir backbone + classifier baru** (ResNet: `stage4`; MobileNetV3: `stage5`, `final_block`, `conv1` classifier). BatchNorm di lapisan beku tetap mode eval. Alasannya: data kecil (±1.500 gambar), jadi fitur umum di lapisan awal dipertahankan dan hanya fitur tingkat tinggi yang disesuaikan.

| Pengaturan | Nilai |
|---|---|
| Input | resize 144×144 → random crop 128×128 (train) / center crop (eval) |
| Augmentasi (train) | random crop, rotasi ±10°, color jitter (brightness/contrast/saturation 0,3; hue 0,05) |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | backbone 3e-4, head 2e-3, jadwal OneCycle (pct_start 0,2) |
| Loss | cross-entropy, label smoothing 0,05 (hanya saat training; loss val/test tanpa smoothing) |
| Epoch / batch | 8 / 32 |
| Pemilihan checkpoint | epoch dengan akurasi validasi tertinggi |
| Ulangan | 3 seed (1, 2, 3) per model |

**4. Evaluasi.** Akurasi, macro-F1, dan loss pada test set (checkpoint terbaik), confusion matrix, jumlah parameter, waktu training, dan **latensi inferensi** (median 60 kali, CPU 1 thread, batch 1).

## Catatan penting

- **Sumber bobot pretrained.** Server tempat eksperimen dijalankan tidak bisa mengakses `download.pytorch.org`, jadi bobot ImageNet diambil dari paket [`pytorchcv`](https://github.com/osmr/imgclsmob) (arsitektur sama, bobot dilatih terpisah dari bobot `torchvision`). Jika Anda ingin memakai bobot `torchvision`, ganti fungsi `build()` di `src/train.py`; hasil bisa sedikit berbeda.
- **Kelas `aexample`** dibuang dari klasifikasi (lihat datasheet).
- **Crop berasal dari anotasi ground-truth**, bukan dari detektor. Hasil ini mengukur kemampuan *klasifikasi*, bukan deteksi+klasifikasi end-to-end.
- Lisensi dataset: **CC BY 4.0** (Roboflow Universe, `brail-nyx-88/crop-aja`); sertakan atribusi jika dataset dipublikasikan ulang. Tentukan sendiri lisensi untuk kode sebelum repo dipublikasikan.
