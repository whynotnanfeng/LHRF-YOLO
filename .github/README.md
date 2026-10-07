# LHRF-YOLO

[![Paper](https://img.shields.io/badge/paper-Forests%202025%2C%2016%2C%201095-8B0000)](https://www.mdpi.com/1999-4907/16/7/1095)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE)

**2.25 M parameters · 5.4 GFLOPs · 87.6% mAP@0.5**

Lightweight forest fire and smoke detection with a hybrid receptive field, built
on the Ultralytics YOLO framework. Three modules target the parts of the problem
that matter for fire and smoke:

- **RMELAN** — Residual Multi-Branch Efficient Layer Aggregation Networks (§2.2.2).
  Fuses 2D selective scan with multi-branch convolution, so the receptive field
  grows without the quadratic cost of self-attention.
- **DEPMD** — Dynamic Enhanced Patch Merge Downsampling (§2.2.3). Reweights
  channels while downsampling, so faint smoke texture survives the reduction.
- **SWF** — Scaling Weighted Fusion (§2.2.4). Learns a per-branch scale when
  merging pyramid levels instead of concatenating them at a fixed ratio.

Ma, Shan, Sui, Wang, Wang. *LHRF-YOLO: A Lightweight Model with Hybrid Receptive
Field for Forest Fire Detection.* **Forests** 2025, *16*(7), 1095.
[Article](https://www.mdpi.com/1999-4907/16/7/1095) ·
[DOI](https://doi.org/10.3390/f16071095)

## Install

```bash
git clone https://github.com/whynotnanfeng/LHRF-YOLO.git
cd LHRF-YOLO
pip install -r requirements.txt
```

A CUDA GPU is required for training: the selective scan in `SSBlock` calls the
CUDA extensions in `selective_scan/`. When Triton is unavailable the code falls
back to an equivalent PyTorch implementation, which is slower but functional.

## Dataset

**Fire-SmokeDataset** — 19,866 images, split 6:3:1 into train 12,083 / val 5,640
/ test 2,143. Two classes: `fire` and `smoke`.

| Mirror | Link |
|--------|------|
| Baidu Netdisk | https://pan.baidu.com/s/1_Xti5AoIER3yZ5hQhToSZQ (code `m58i`) |
| Google Drive | https://drive.google.com/file/d/1VFYYXHbzDTtgjTUU8cl6qNgi8ZahxGSe/view?usp=sharing |
| QuarkDrive | https://pan.quark.cn/s/c13d3a6251c0 (code `4aii`) |

```text
FSDataset/
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Point `path` in `data_fire-smoke.yaml` at the dataset root, then check the layout
before a long run:

```bash
yolo checks data=data_fire-smoke.yaml
```

Labels use YOLO format: `class x_center y_center width height`, normalized to
`[0, 1]`. Details in [`docs/DATASET.md`](docs/DATASET.md).

## Train

The paper trains with AdamW at `lr0=0.001`; the Ultralytics defaults are SGD at
`lr0=0.01`, so pass the paper's values explicitly:

```bash
yolo train model=LHRF.yaml data=data_fire-smoke.yaml \
    epochs=100 imgsz=640 batch=64 workers=32 \
    optimizer=AdamW lr0=0.001 momentum=0.937
```

Check the parameter count against the published 2.25 M without a dataset or a
GPU:

```bash
python scripts/smoke_test.py --paper
```

## Citation

```bibtex
@article{ma2025lhrfyolo,
  title   = {LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field
             for Forest Fire Detection},
  author  = {Ma, Yifan and Shan, Weifeng and Sui, Yanwei and Wang, Mengyu and Wang, Maofa},
  journal = {Forests},
  volume  = {16},
  number  = {7},
  pages   = {1095},
  year    = {2025},
  doi     = {10.3390/f16071095}
}
```

Full documentation: [README.md](README.md) ·
Dataset schema: [`docs/DATASET.md`](docs/DATASET.md)

## License

AGPL-3.0, inherited from Ultralytics YOLO. The article is © 2025 by the authors
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
