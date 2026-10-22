from ultralytics import YOLO


def main():
    # Build LHRF-YOLO from the model definition.
    model = YOLO("lhrf.yaml")
    model.train(
        data="data_fire-smoke.yaml",
        epochs=100,
        batch=64,
        imgsz=640,
        workers=8,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        pretrained=False,
        seed=42,
        deterministic=True,
        project="runs/train",
        name="lhrf_yolo",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
