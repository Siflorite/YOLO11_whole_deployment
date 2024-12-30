from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./yolo11n.torchscript")
    source="F:/YOLO/multicell/multicell2.mp4"
    model.predict(source, save=True, device="0")