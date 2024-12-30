from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data="F:/YOLO/yolo11/datasets/multicell/data.yaml",  # path to dataset YAML
        epochs=200,  # number of training epochs
        batch=4,
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    metrics = model.val()
    results = model("F:/YOLO/multicell/multicell1.avi")
    # path = model.export(format="onnx")  # return path to exported model