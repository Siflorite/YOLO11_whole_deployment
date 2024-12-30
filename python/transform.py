from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('./runs/detect/train8/weights/best.pt')
    model.export(format='torchscript')
    model.export(format='onnx')