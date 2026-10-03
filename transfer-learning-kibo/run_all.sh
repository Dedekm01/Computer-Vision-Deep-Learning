#!/usr/bin/env bash
# Menjalankan seluruh pipeline: crop -> train (3 seed) -> chart.
# Letakkan hasil ekspor Roboflow (YOLOv8) di data/yolov8 terlebih dahulu.
set -e
python src/make_crops.py --src data/yolov8 --out data/cls
python src/dataset_figures.py --data data/cls
for s in 1 2 3; do python src/train.py --seed $s --epochs 8; done
python src/charts.py
