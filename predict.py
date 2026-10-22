from ultralytics import YOLO


def main():
    model = YOLO("runs/train/lhrf_yolo/weights/best.pt")
    model.predict(source="path/to/image_or_folder", imgsz=640, conf=0.5, save=True)


if __name__ == "__main__":
    main()
