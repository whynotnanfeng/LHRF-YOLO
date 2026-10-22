from ultralytics import YOLO


def main():
    model = YOLO("runs/train/lhrf_yolo/weights/best.pt")
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)


if __name__ == "__main__":
    main()
