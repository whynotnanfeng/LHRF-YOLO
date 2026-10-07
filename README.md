# LHRF-YOLO

A lightweight YOLO variant for forest fire and smoke detection, built on the
[Ultralytics](https://github.com/ultralytics/ultralytics) framework.

## Paper

This repository accompanies the following publication:

> **LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for Forest Fire Detection**
>
> Yifan Ma, Weifeng Shan, Yanwei Sui, Mengyu Wang, Maofa Wang
> *Forests* **2025**, *16*(7), 1095.
>
> Article: https://www.mdpi.com/1999-4907/16/7/1095
> DOI: https://doi.org/10.3390/f16071095

### Modules

The three contributions of the paper are implemented in
[`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py) and wired
together in [`LHRF.yaml`](LHRF.yaml).

| Module | Class | Purpose |
| --- | --- | --- |
| RMELAN | `block.py: RMELAN` | Multi-branch aggregation combining convolutional branches with 2D selective scan for a large receptive field at linear complexity |
| DEPMD | `block.py: DEPMD` | Patch-merge downsampling that reorganizes four interleaved sub-samples and re-weights channels, preserving fine smoke texture |
| SWF | `block.py: SWF` | Scale weighted fusion that learns per-branch weights for multi-scale feature fusion |

Supporting classes: `SSBlock`, `CGLU`, `DynamicSparseGate`, `LayerNorm2d`.

## Requirements

- Python >= 3.8
- PyTorch >= 1.8.0
- A CUDA-capable GPU is recommended for training and inference.

## Installation

```bash
git clone https://github.com/whynotnanfeng/LHRF.git
cd LHRF

# Install dependencies
pip install -r requirements.txt

# Or install in editable mode
pip install -e .
```

The selective scan operators use Triton and the CUDA kernels in
[`selective_scan/`](selective_scan). When Triton is unavailable the code falls
back to an equivalent PyTorch implementation, which is slower but functionally
equivalent.

## Dataset

The dataset used in the paper is available from the following mirrors:

- **Baidu Netdisk**: [link](https://pan.baidu.com/s/1_Xti5AoIER3yZ5hQhToSZQ) (code `m58i`)
- **Google Drive**: [link](https://drive.google.com/file/d/1VFYYXHbzDTtgjTUU8cl6qNgi8ZahxGSe/view?usp=sharing)
- **QuarkDrive**: [link](https://pan.quark.cn/s/c13d3a6251c0) (code `4aii`)

Organize the data as expected by `data_fire-smoke.yaml`:

```text
FSDataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Two classes are used: `fire` and `smoke`. Adjust the `path` field in
`data_fire-smoke.yaml` to point at your local dataset directory.

## Usage

### Training

```bash
# Train with the default configuration
yolo train model=LHRF.yaml data=data_fire-smoke.yaml epochs=100 imgsz=640

# Resume from the last checkpoint
yolo train model=runs/detect/train/weights/last.pt resume=True
```

Key training arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `epochs` | `100` | Number of training epochs |
| `imgsz` | `640` | Training image size |
| `batch` | auto | Batch size; auto-scales to available GPU memory |
| `device` | auto | Training device, e.g. `0` for GPU 0 or `cpu` |
| `workers` | `8` | Dataloader worker processes |
| `patience` | `100` | Early stopping patience in epochs |
| `seed` | `0` | Random seed for reproducibility |
| `project` | `runs/detect` | Output directory root |
| `name` | `train` | Experiment name |

### Inference

```bash
# Predict on an image
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

# Predict on a video
yolo predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4
```

### Validation

```bash
yolo val model=runs/detect/train/weights/best.pt data=data_fire-smoke.yaml
```

### Export

```bash
# Export to ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx

# Export to TensorRT
yolo export model=runs/detect/train/weights/best.pt format=engine
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("LHRF.yaml")

# Train
model.train(data="data_fire-smoke.yaml", epochs=100, imgsz=640)

# Predict
results = model.predict(source="path/to/image.jpg")
```

## License

This project builds on Ultralytics YOLO, which is licensed under AGPL-3.0. See
[`LICENSE`](LICENSE) for the full terms.

The Fire-SmokeDataset is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); please cite the
original authors when reusing it.
