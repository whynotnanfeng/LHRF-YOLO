# LHRF-YOLO

[![Paper](https://img.shields.io/badge/paper-Forests%202025%2C%2016%2C%201095-8B0000)](https://www.mdpi.com/1999-4907/16/7/1095)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-FSDataset-blue)](docs/DATASET.md)

Lightweight forest fire and smoke detection for edge and UAV deployment, with a
hybrid receptive field.

**2.25 M parameters · 5.4 GFLOPs · 87.6% mAP@0.5**

Official PyTorch implementation of *LHRF-YOLO: A Lightweight Model with Hybrid
Receptive Field for Forest Fire Detection* — Ma, Shan, Sui, Wang, Wang,
*Forests* **2025**, *16*(7), 1095 · DOI [10.3390/f16071095](https://doi.org/10.3390/f16071095)

---

## What the paper adds

Flames spread dynamically, smoke is semi-transparent, and both sit in cluttered
forest backgrounds, so the features a detector needs are hard to extract while the
model still has to run on power-limited hardware. Each module targets one part of
that problem.

| Module | Paper | Problem it solves | Code |
|---|---|---|---|
| **RMELAN** | §2.2.2 | Local detail and global context are modeled separately: convolution alone gives a limited receptive field, and self-attention over a flattened feature map is quadratic in the number of tokens. | [`block.py` → `RMELAN`](ultralytics/nn/modules/block.py) |
| **DEPMD** | §2.2.3 | Strided downsampling and plain patch merging discard fine smoke texture, which is precisely what faint targets depend on. | [`block.py` → `DEPMD`](ultralytics/nn/modules/block.py) |
| **SWF** | §2.2.4 | Concatenating pyramid levels treats shallow and deep features as interchangeable, so deep semantic layers get diluted. | [`block.py` → `SWF`](ultralytics/nn/modules/block.py) |
| **Mish** | §2.2.5 | SiLU underweights the faint gradients of semi-transparent smoke boundaries and flame edges. | activation used throughout the backbone |

Supporting classes: `SSBlock` and `CGLU` (the selective scan inside RMELAN),
`DynamicSparseGate` (channel sparsification inside RMELAN), `LayerNorm2d`.

RMELAN fuses 2D selective scan (SS2D) with residual multi-branch convolution,
giving a large receptive field at linear complexity. DEPMD merges four
interleaved sub-samples and re-weights them channel-wise. SWF learns a per-branch
scale rather than merging features at a fixed ratio.

---

## Architecture

24 layers from `LHRF.yaml` at scale `n`, two classes:

```
 0Conv              P1/2
 1  DEPMD           P2/4
 2  C3k2
 3  DEPMD                     P3/8
 4  C3k2
 5  DEPMD                     P4/16
 6  C3k2
 7  DEPMD                     P5/32
 8  RMELAN                    hybrid receptive field: SS2D + multi-branch
 9  DEPMD
10  RMELAN
11  SPPF
12  C2PSA
13  Upsample                  P5 -> P4
14  SWF                       learned scale per branch
15  C3k2                                P3/8  (small)
16  Upsample                  P3 -> P4
17  SWF
18  C3k2                                P4/16 (medium)
19  Conv                stride 2
20  SWF
21  C3k2
22  Conv                stride 2
23  RMELAN + SWF                     P5/32 (large)
24  Detect                   cat(P3, P4, P5) -> 2 classes
```

### Complexity check

Measured against the figures reported in the paper:

| Metric | Paper | Measured | |
|---|---|---|---|
| Parameters | 2.25 M | **2.2564 M** | within 0.3% |
| Reduction vs YOLOv11n | 12.8% | **12.5%** | baseline 2.58 M |

Reproduce with `python scripts/smoke_test.py --paper`. The script builds the model
from `LHRF.yaml` at `nc=2` (the two dataset classes) and compares the parameter
count against the published value.

---

## Installation

Python >= 3.8, PyTorch >= 1.8. A CUDA GPU is recommended: the selective scan
kernels in `selective_scan/` need Triton and a CUDA toolchain to build.

```bash
git clone https://github.com/whynotnanfeng/LHRF-YOLO.git
cd LHRF-YOLO

python -m venv .venv && source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
numpy>=1.23.0
matplotlib>=3.3.0
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0
psutil
py-cpuinfo
pandas>=1.1.4
seaborn>=0.11.0
ultralytics-thop>=2.0.0
timm>=0.9.0
einops>=0.6.0
```
</details>

When Triton is unavailable, `block.py` falls back to an equivalent PyTorch
implementation of the selective scan. It is slower but functionally equivalent,
so a CPU-only install still runs.

---

## Dataset

**Fire-SmokeDataset** — 19,866 images, split 6:3:1 into train 12,083 / val 5,640
/ test 2,143. Two classes: `fire` and `smoke`.

| Mirror | Link |
|--------|------|
| Baidu Netdisk | https://pan.baidu.com/s/1_Xti5AoIER3yZ5hQhToSZQ (code `m58i`) |
| Google Drive | https://drive.google.com/file/d/1VFYYXHbzDTtgjTUU8cl6qNgi8ZahxGSe/view?usp=sharing |
| QuarkDrive | https://pan.quark.cn/s/c13d3a6251c0 (code `4aii`) |

Extract into this layout and point `path` in `data_fire-smoke.yaml` at the root:

```text
FSDataset/
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Label files are in YOLO format: one line per object,
`class x_center y_center width height`, normalized to `[0, 1]`. Full field
reference in [`docs/DATASET.md`](docs/DATASET.md).

---

## Usage

### Training

```bash
# Train with the published configuration
yolo train model=LHRF.yaml data=data_fire-smoke.yaml epochs=100 imgsz=640

# Resume the last run
yolo train model=runs/detect/train/weights/last.pt resume=True
```

Key arguments:

| Argument | Default | Meaning |
|---|---|---|
| `epochs` | `100` | Training epochs |
| `imgsz` | `640` | Input resolution |
| `batch` | auto | Batch size; auto-scales to available GPU memory |
| `device` | auto | `0` for GPU 0, `cpu`, or `0,1` for multi-GPU |
| `workers` | `8` | Dataloader workers |
| `patience` | `100` | Early-stopping patience in epochs |
| `optimizer` | auto | Optimizer; `auto` selects one for this model |
| `seed` | `0` | Random seed |
| `project` / `name` | `runs/detect` / `train` | Output directory and experiment name |

### Inference

```bash
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
yolo predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4
```

### Validation

```bash
yolo val model=runs/detect/train/weights/best.pt data=data_fire-smoke.yaml
```

### Export

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
yolo export model=runs/detect/train/weights/best.pt format=engine
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("LHRF.yaml")
model.train(data="data_fire-smoke.yaml", epochs=100, imgsz=640)

results = model.predict(source="path/to/image.jpg")
```

### Verifying the setup

```bash
python scripts/smoke_test.py --paper   # parameter count vs the published figure
python scripts/smoke_test.py --flops   # MACs and GFLOPs (needs the CUDA extensions)
```

---

## Repository layout

```text
LHRF-YOLO/
├── LHRF.yaml                 # model definition: RMELAN, DEPMD, SWF and the P3-P5 head
├── data_fire-smoke.yaml      # dataset paths and class names
├── requirements.txt
├── scripts/
│   └── smoke_test.py         # builds the model, checks parameters against the paper
├── docs/
│   └── DATASET.md            # dataset layout, label format, access mirrors
├── selective_scan/           # Triton and CUDA kernels for the selective scan
├── tests/
└── ultralytics/
    └── nn/modules/
        └── block.py# RMELAN, DEPMD, SWF, SSBlock, CGLU, DynamicSparseGate
```

This repository is a fork of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
The paper's contributions live in `ultralytics/nn/modules/block.py` and
`LHRF.yaml`; the rest of the package is kept close to upstream so that merging
later upstream changes stays straightforward.

---

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

---

## License

Code is released under the AGPL-3.0 License, inherited from Ultralytics YOLO. See
[`LICENSE`](LICENSE).

The article is © 2025 by the authors under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Fire-SmokeDataset
images are not included here; see [`docs/DATASET.md`](docs/DATASET.md).
