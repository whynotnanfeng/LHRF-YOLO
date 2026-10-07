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

A smoke plume is often annotated as a box around the visible smoke rather than
a tight mask, so boxes for the same fire event can overlap.

Check the layout before a long run:

```bash
yolo checks data=data_fire-smoke.yaml
```

The dataset checks scan for missing or corrupt images, verify the label files
parse, and report the class distribution.

---

## Usage

### Training

The paper trains with the hyperparameter configuration in its Table 3, on a
Tesla V100. The defaults shipped with Ultralytics differ, so pass the paper's
values explicitly:

```bash
yolo train model=LHRF.yaml data=data_fire-smoke.yaml \
    epochs=100 imgsz=640 batch=64 workers=32 \
    optimizer=AdamW lr0=0.001 momentum=0.937 weight_decay=0.0005
```

| Parameter | Paper (Table 3) | Ultralytics default | Note |
|---|---|---|---|
| Input size | 640 x 640 | 640 | same |
| Epochs | 100 | 100 | same |
| Batch size | 64 | 16 | paper value needs a large GPU; lower it if memory is tight |
| Workers | 32 | 8 | |
| Optimizer | **AdamW** | `auto` (SGD) | **must be set explicitly** |
| Learning rate (`lr0`) | **0.001** | 0.01 | **must be set explicitly**; 0.01 is the SGD default |
| Momentum (`momentum`) | 0.937 | 0.937 | same; this is Adam beta1 |
| Weight decay | 0.0005 | 0.0005 | see the note below |

> `momentum=0.937` is the Adam beta1, not SGD momentum. Leave it as shown.
> The paper does not report a learning-rate schedule, so the default linear
> schedule with a 3-epoch warmup is used here.

**Known deviation.** Ultralytics builds AdamW with a hard-coded
`weight_decay=0.0` (`ultralytics/engine/trainer.py`), so the paper's 0.0005
does not take effect through the `weight_decay` argument. Passing it is
harmless, but the run will use zero weight decay. Reproducing the published
number exactly requires patching that optimizer call.

Other useful arguments:

| Argument | Default | Meaning |
|---|---|---|
| `device` | auto | `0` for GPU 0, `cpu`, or `0,1` for multi-GPU |
| `patience` | `100` | Early-stopping patience in epochs |
| `seed` | `0` | Random seed |
| `project` / `name` | `runs/detect` / `train` | Output directory and experiment name |
| `resume` | `False` | Resume the last run from its checkpoint |

```bash
# Resume an interrupted run
yolo train model=runs/detect/train/weights/last.pt resume=True

# Evaluate on the test split instead of val
yolo val model=runs/detect/train/weights/best.pt data=data_fire-smoke.yaml split=test
```

Checkpoints and logs are written to `runs/detect/train/`, with the resolved
configuration saved alongside as `args.yaml`. Watch progress with:

```bash
pip install tensorboard          # optional, not in requirements.txt
tensorboard --logdir runs/detect/train
```

### Inference

```bash
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
yolo predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4
```

### Validation

```bash
yolo val model=runs/detect/train/weights/best.pt data=data_fire-smoke.yaml

# Report metrics on the test split
yolo val model=runs/detect/train/weights/best.pt data=data_fire-smoke.yaml split=test
```

`split` selects which partition of `data_fire-smoke.yaml` is evaluated
(`val`, `test` or `train`); it defaults to `val`.

### Export

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
yolo export model=runs/detect/train/weights/best.pt format=engine
```

### Python API

The command line and the Python API take the same arguments. Use this form when
the run needs to be scripted, or to pass the paper's hyperparameters:

```python
from ultralytics import YOLO

model = YOLO("LHRF.yaml")            # build from the model YAML
model.train(
    data="data_fire-smoke.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    workers=32,
    optimizer="AdamW",
    lr0=0.001,
    momentum=0.937,
    weight_decay=0.0005,
)

results = model.predict(source="path/to/image.jpg")
metrics = model.val(data="data_fire-smoke.yaml")
```

To load a trained checkpoint instead of the YAML:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

> A CUDA GPU is required for training and for `YOLO(...)` to build the model,
> because the selective scan in `SSBlock` calls the CUDA extensions. On a
> CPU-only install, `scripts/smoke_test.py` still verifies the parameter
> count.

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
