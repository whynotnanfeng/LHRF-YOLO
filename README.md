# LHRF-YOLO
## 📄 Paper

This repository is the official implementation of the following paper. The model
architecture in [`LHRF.yaml`](LHRF.yaml) and the custom modules in
[`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py) correspond
directly to the method described in the paper.

**LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for Forest Fire Detection**

> Yifan Ma, Weifeng Shan, Yanwei Sui, Mengyu Wang, Maofa Wang
> *Forests*, 2025, **16**(7), 1095.
> [[Article page]](https://www.mdpi.com/1999-4907/16/7/1095) · [[PDF]](https://www.mdpi.com/1999-4907/16/7/1095/pdf) · [[DOI]](https://doi.org/10.3390/f16071095)

### Method ↔ Code Map

The three contributions of LHRF-YOLO map onto the code as follows.

| Paper module | Full name | Code location | Role |
|---|---|---|---|
| **RMELAN** | Residual Multi-Branch Efficient Layer Aggregation Network | `block.py: class RMELAN` | Hybrid receptive field extraction — combines 2D selective scan (SS2D) with a residual multi-branch structure to model local detail and global context at linear complexity |
| **DEPMD** | Dynamic Enhanced Patch Merge Downsampling | `block.py: class DEPMD` | Feature reorganization + channel-wise dynamic weighting, preserving fine smoke texture while reducing spatial resolution |
| **SWF** | Scale Weighted Fusion | `block.py: class SWF` | Adaptive scale weight allocation for multi-scale feature fusion, avoiding information dilution |
| **Mish** | Mish activation | replaces SiLU in the backbone | Improves capture of flame edges and faint, semi-transparent smoke textures |

Supporting implementations:

- `SSBlock` / `CGLU` — the selective-scan block used inside RMELAN
- `DynamicSparseGate` — the dynamic gating mechanism that allocates channels in RMELAN
- `selective_scan/` — the CUDA selective-scan kernels

### Reported Results

On the self-constructed **Fire-SmokeDataset**, compared with the YOLOv11n baseline:

| Metric | Value | Change |
|---|---|---|
| Parameters | 2.25 M | −12.8% |
| GFLOPs | 5.4 | −14.3% |
| mAP50 | 87.6% | improved |

The model also shows leading generalization on the cross-scenario **M4SFWD**
dataset, and has been deployed on NVIDIA Jetson edge platforms.

### Citation

If you use this work in your research, please cite the original paper:

```bibtex
@article{ma2025lhrf,
  title   = {LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for Forest Fire Detection},
  author  = {Ma, Yifan and Shan, Weifeng and Sui, Yanwei and Wang, Mengyu and Wang, Maofa},
  journal = {Forests},
  volume  = {16},
  number  = {7},
  pages   = {1095},
  year    = {2025},
  doi     = {10.3390/f16071095}
}
```

> The dataset is distributed under CC BY 4.0; please cite the original authors
> when reusing Fire-SmokeDataset.

## 📁 Dataset Download

The dataset used in this project is available via the following cloud storage links:

- 🔗 **Baidu Netdisk**: [Download here](https://pan.baidu.com/s/1_Xti5AoIER3yZ5hQhToSZQ) (Extraction code: `m58i`)  
- 🌍 **Google Drive**: [Download here](https://drive.google.com/file/d/1VFYYXHbzDTtgjTUU8cl6qNgi8ZahxGSe/view?usp=sharing)  
- 📦 **Alternative Link**: [QuarkDrive](https://pan.quark.cn/s/c13d3a6251c0) (Extraction code: `4aii`)  

> ⚠️ Note: Please choose the appropriate download option based on your region. Some links may have slower access speeds depending on your network location.

## 🚀 Deployment Guide

### Requirements

- Python >= 3.8
- CUDA GPU (recommended for accelerated training and inference)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/whynotnanfeng/LHRF.git
cd LHRF

# Install dependencies (choose one)
pip install -r requirements.txt       # Standard installation
# OR
pip install -e .                      # Editable mode for development
```

### Quick Start

**Train Model**
```bash
yolo train model=LHRF.yaml data=data_fire-smoke.yaml epochs=100 imgsz=640
```

**Inference**
```bash
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
```

**Export Model**
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO("LHRF.yaml")

# Train
model.train(data="data_fire-smoke.yaml", epochs=100, imgsz=640)

# Inference
results = model.predict(source="path/to/image.jpg")
```
