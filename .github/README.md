# LHRF-YOLO

Official implementation of **LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for
Forest Fire Detection**.

This repository provides the model definition, training configuration and inference code for the
paper's three contributions: the RMELAN aggregation module, the DEPMD downsampling module and the
SWF fusion module, built on top of the Ultralytics YOLO framework.

## Paper

> Yifan Ma, Weifeng Shan, Yanwei Sui, Mengyu Wang, Maofa Wang
> **LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for Forest Fire Detection**
> *Forests* **2025**, *16*(7), 1095.
>
> Article: https://www.mdpi.com/1999-4907/16/7/1095
> DOI: https://doi.org/10.3390/f16071095

## Installation

```bash
git clone https://github.com/whynotnanfeng/LHRF-YOLO.git
cd LHRF-YOLO
pip install -r requirements.txt
```

See [README.md](README.md) for dataset layout, training arguments and usage examples.

## License

AGPL-3.0. See [LICENSE](LICENSE).
