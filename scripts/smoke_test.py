"""Reproduce the complexity figures reported in the LHRF-YOLO paper.

The paper states that LHRF-YOLO has 2.25 M parameters and 5.4 GFLOPs, a 12.8%
and 14.3% reduction against the YOLOv11n baseline. This script builds the model
from ``LHRF.yaml`` and measures the parameter count so the claim can be checked
without a dataset or trained weights.

FLOPs are reported as 2 x MACs, the convention used in the paper.

Usage:
    python scripts/smoke_test.py              # parameter count and a CPU forward pass
    python scripts/smoke_test.py --flops      # also measure MACs/FLOPs with thop
    python scripts/smoke_test.py --paper      # compare against the paper's numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reported in "LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for
# Forest Fire Detection", Forests 2025, 16(7), 1095.
PAPER_PARAMS = 2_250_000
PAPER_GFLOPS = 5.4
PAPER_BASELINE_PARAMS = 2.58e6  # YOLOv11n, for the 12.8% reduction
TOLERANCE = 0.02  # accept a 2% deviation from the reported figures


def build_model(nc: int = 2, scale: str = "n"):
    """Build LHRF-YOLO from its YAML description.

    Args:
        nc: Number of classes. The dataset has two (fire, smoke), which is what the
            paper's 2.25 M figure corresponds to.
        scale: Model scale from the YAML ``scales`` block.
    """
    from ultralytics.nn.tasks import parse_model

    cfg = yaml.safe_load((REPO_ROOT / "LHRF.yaml").read_text(encoding="utf-8"))
    cfg["nc"] = nc
    cfg["scale"] = scale
    model, _ = parse_model(cfg, ch=3, verbose=False)
    return model


def count_params(model) -> int:
    """Return the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, imgsz: int = 640):
    """Return (MACs, GFLOPs) at the given input size, with GFLOPs = 2 x MACs.

    Requires ``thop``. The selective scan needs the CUDA extensions, so this is
    skipped when they are unavailable.
    """
    import torch
    from thop import profile

    macs, _ = profile(model, inputs=(torch.randn(1, 3, imgsz, imgsz),), verbose=False)
    return macs, 2 * macs


def forward_check(model, imgsz: int = 640) -> None:
    """Run CPU forward passes over the custom modules.

    Exercises DEPMD and SWF, which are pure PyTorch. SSBlock is skipped because it
    calls the selective scan CUDA extensions.
    """
    import torch

    seen = set()
    for _, m in model.named_modules():
        kind = type(m).__name__
        if kind not in {"DEPMD", "SWF"} or kind in seen:
            continue
        seen.add(kind)
        size = imgsz // 4
        if kind == "DEPMD":
            # DEPMD expands its input by 4 internally: Conv2d(4*dim -> out_dim).
            dim = m.hidden // 4
            x = torch.randn(1, dim, size, size)
            print(f"  DEPMD              {tuple(m(x).shape)}")
        else:  # SWF concatenates two features along channels
            x = torch.randn(1, 32, size, size)
            print(f"  SWF                {tuple(m([x, x]).shape)}")
    if not seen:
        print("  no custom modules found in the model")


def main() -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc", type=int, default=2, help="number of classes (default: 2)")
    parser.add_argument("--imgsz", type=int, default=640, help="input size (default: 640)")
    parser.add_argument("--flops", action="store_true", help="also measure FLOPs with thop")
    parser.add_argument("--paper", action="store_true", help="compare against the paper figures")
    args = parser.parse_args()

    print(f"Building LHRF-YOLO from LHRF.yaml (nc={args.nc}, scale=n, imgsz={args.imgsz})")
    model = build_model(nc=args.nc)
    params = count_params(model)
    print(f"  parameters: {params:,} ({params / 1e6:.4f} M)")

    print("Running a CPU forward pass")
    try:
        forward_check(model, imgsz=args.imgsz)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  forward check skipped: {type(exc).__name__}: {exc}")

    if args.flops:
        try:
            macs, gflops = measure_flops(model, args.imgsz)
            print(f"  MACs: {macs / 1e9:.3f} G")
            print(f"  GFLOPs (2 x MACs): {gflops / 1e9:.3f} G")
        except Exception as exc:  # noqa: BLE001
            print(f"  FLOPs measurement unavailable: {type(exc).__name__}: {exc}")

    if args.paper:
        print("\nComparison with the paper")
        dev = abs(params - PAPER_PARAMS) / PAPER_PARAMS
        status = "OK" if dev <= TOLERANCE else "MISMATCH"
        print(f"  parameters: {params / 1e6:.4f} M vs {PAPER_PARAMS / 1e6:.2f} M  [{status}]")
        reduction = (1 - params / PAPER_BASELINE_PARAMS) * 100
        print(f"  reduction vs YOLOv11n: {reduction:.1f}% (paper reports 12.8%)")
        if dev > TOLERANCE:
            print(f"  deviation {dev * 100:.2f}% exceeds tolerance {TOLERANCE * 100:.0f}%")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
