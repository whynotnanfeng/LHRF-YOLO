# LHRF-YOLO

[![Paper](https://img.shields.io/badge/paper-Forests%202025%2C%2016%2C%201095-8B0000)](https://www.mdpi.com/1999-4907/16/7/1095)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE)

**2.25 M parameters · 5.4 GFLOPs · 87.6% mAP@0.5**

Lightweight forest fire and smoke detection with a hybrid receptive field, built
on the Ultralytics YOLO framework. Three modules target the parts of the problem
that matter for fire and smoke: **RMELAN** fuses 2D selective scan with
multi-branch convolution for a large receptive field at linear complexity,
**DEPMD** reweights channels while downsampling so faint smoke texture survives,
and **SWF** learns a per-branch scale when merging pyramid levels.

Ma, Shan, Sui, Wang, Wang. *LHRF-YOLO: A Lightweight Model with Hybrid Receptive
Field for Forest Fire Detection.* **Forests** 2025, *16*(7), 1095.
[Article](https://www.mdpi.com/1999-4907/16/7/1095) ·
[DOI](https://doi.org/10.3390/f16071095)

```bash
git clone https://github.com/whynotnanfeng/LHRF-YOLO.git
cd LHRF-YOLO
pip install -r requirements.txt

yolo train model=LHRF.yaml data=data_fire-smoke.yaml epochs=100 imgsz=640
```

See [README.md](README.md) for the full documentation.
