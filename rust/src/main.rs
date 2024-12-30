use tch::Device;
use std::path::Path;
use ort;

mod onnx_predict;
mod libtorch_predict;
mod bbox;
mod utils;

use utils::load_cuda_dll;
use libtorch_predict::YOLOv8;
use onnx_predict::Yolo11Onnx;

fn libtorch_func() {
    load_cuda_dll();
    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let weights_path = "F:/YOLO/yolo11/rust/yolo_self/weights/yolo11n.torchscript";
    let weights = Path::new(weights_path);
    let (h, w) = (720, 720);
    let conf_threshold = 0.25;
    let iou_threshold = 0.45;
    let top_k = 100;
    let mut yolo = YOLOv8::new(weights, h, w, conf_threshold, iou_threshold, top_k, device).unwrap();
    let img = yolo.preprocess("F:/YOLO/yolo11/rust/yolo_self/test/sampleimg170.jpg").unwrap();
    let output = yolo.predict(&img);
    println!("output: {:?}, size: {}", output, output.len());

    let img = image::open("F:/YOLO/yolo11/rust/yolo_self/test/sampleimg170.jpg").unwrap();
    yolo.show(img.into_rgb8(), &output);
}

fn onnx_func() -> Result<(), ort::Error>{
    let weights_path = "F:/YOLO/yolo11/rust/yolo_self/weights/yolo11n.onnx";
    let weights = Path::new(weights_path);
    let (h, w) = (720, 720);
    let conf_threshold = 0.25;
    let iou_threshold = 0.45;
    let top_k = 100; 
    let mut yolo = Yolo11Onnx::new(weights, h, w, conf_threshold, iou_threshold, top_k).unwrap();
    let img = yolo.preprocess("F:/YOLO/yolo11/rust/yolo_self/test/sampleimg170.jpg").unwrap();
    let output = yolo.predict(img)?;
    println!("output: {:?}, size: {}", output, output.len());
    let img = image::open("F:/YOLO/yolo11/rust/yolo_self/test/sampleimg170.jpg").unwrap();
    yolo.show(img.into_rgb8(), &output);
    Ok(())
}

fn main() {
    libtorch_func();
    onnx_func().unwrap();
}
