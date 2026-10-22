from ultralytics import YOLO


def main():
    model = YOLO("runs/train/lhrf_yolo/weights/best.pt")
    model.val(data="data_fire-smoke.yaml", imgsz=640, batch=64, device=0)


if __name__ == "__main__":
    main()
