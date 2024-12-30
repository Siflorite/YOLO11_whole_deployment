# YOLO11_whole_deployment
This is a throughout YOLO11 deployment mainly for my personal use. Uses PyTorch to train yolo11n models, and libtorch/onnxruntime under cpp/Rust for higher efficiency deployment.<br>
The CPP project is based on [Ultralytic's YOLOv8 LibTorch CPP Inference](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-LibTorch-CPP-Inference).
The Rust project is based on [this](https://github.com/kingzcheung/yolov8/blob/main/src/tch.rs) for libtorch, and the onnx part is modified to meet onnxrumtime's need based on [ort's example of yolov8](https://github.com/pykeio/ort/tree/main/examples/yolov8).
