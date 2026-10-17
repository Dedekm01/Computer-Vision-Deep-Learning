# Hasil Percobaan Deteksi Bola

[Halaman utama](../README.md) · [Laporan percobaan 01](percobaan_01/README.md)

![Ringkasan E1 dan E3](../docs/images/ringkasan_e1_e3.png)

| Percobaan | Model terpilih | mAP50–95 validasi | mAP50–95 test | Recall test |
|---|---|---:|---:|---:|
| [percobaan_01](percobaan_01/README.md) | E3 · input 640 | 72,46% | 71,34% | 98,48% |

Percobaan membandingkan input 416 dan 640 pada YOLO26n dengan dataset serta pengaturan training yang sama. Model dipilih dari validasi, lalu E3 dievaluasi pada 132 gambar test.

[Grafik E1](percobaan_01/E1_pretrained/results.png) · [Grafik E3](percobaan_01/E3_tuning/results.png) · [CSV perbandingan](percobaan_01/hasil_praktik_4222401050.csv) · [CSV test](percobaan_01/hasil_test_4222401050.csv)
