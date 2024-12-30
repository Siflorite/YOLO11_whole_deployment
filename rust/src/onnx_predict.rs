use image::{GenericImageView, imageops::FilterType};
use ort::{    
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},    
    session::{Session, SessionOutputs},
}; 
use std::{path::Path, time::Instant};
use ndarray::{Array, Axis, s};
use image::{Rgb, ImageBuffer};
use imageproc::rect::Rect;
use ab_glyph::{FontRef, PxScale};

use crate::bbox::{Bbox, iou};
use crate::utils::DROPLET_CLASS_LABELS;

pub struct Yolo11Onnx {
    model: Session,
    h: u32,
    w: u32,
    conf_threshold: f64,
    iou_threshold: f64,
    top_k: u32,
}

impl Yolo11Onnx {
    pub fn new(
        weights: &Path,
        h: u32,
        w: u32,
        conf_threshold: f64,
        iou_threshold: f64,
        top_k: u32,
    ) -> Result<Yolo11Onnx, ort::Error> {
        let model = Session::builder()?
            .with_execution_providers([
                // Prefer TensorRT over CUDA.
                TensorRTExecutionProvider::default().build(), 
                CUDAExecutionProvider::default().build(), 
            ])?
            .commit_from_file(weights)?;
        Ok(
            Yolo11Onnx {
                model,
                h,
                w,
                conf_threshold,
                iou_threshold,
                top_k,
            }
        )  
    }

    pub fn preprocess(&mut self, image_path: &str) -> Result<Array<f32, ndarray::Dim<[usize; 4]>>, ort::Error> {
        let original_img = image::open(image_path).unwrap();
	    (self.w, self.h) = (original_img.width(), original_img.height());
	    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
	    let mut input = Array::zeros((1, 3, 640, 640));
	    for pixel in img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }
        Ok(input)
    }

    pub fn predict(&self, input: Array<f32, ndarray::Dim<[usize; 4]>>) -> Result<Vec<Bbox>, ort::Error> {
        let start_time = Instant::now();
        let outputs: SessionOutputs = self.model.run(ort::inputs!["images" => input.view()]?)?;
        let elapsed_time = start_time.elapsed();
        println!("ONNX inference time:{} ms", elapsed_time.as_millis());

        let start_time = Instant::now();
        let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();
        let result = self.non_max_suppression(output);
        let elapsed_time = start_time.elapsed();
        println!("ONNX nms time:{} ms", elapsed_time.as_millis());
        Ok(result)
    }

    fn non_max_suppression(&self, output: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>) -> Vec<Bbox>{
        let output = output.slice(s![.., .., 0]);
        let nclasses = output.shape()[1] - 4;
        let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                // skip bounding box coordinates
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < self.conf_threshold as f32{
                continue;
            }
            let bbox = Bbox {
                xmin: (row[0] - row[2] / 2.) as f64,
                ymin: (row[1] - row[3] / 2.) as f64,
                xmax: (row[0] + row[2] / 2.) as f64,
                ymax: (row[1] + row[3] / 2.) as f64,
                confidence: prob as f64,
                cls_index: class_id as i64
            };
            bboxes[class_id as usize].push(bbox);
        }
        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                    if iou > self.iou_threshold {
                        drop = true;
                        break;
                    }
                }
                if !drop {
                    bboxes_for_class.swap(current_index, index);
                    current_index += 1;
                }
            }
            bboxes_for_class.truncate(current_index);
        }
        let mut result = vec![];
        let mut count = 0;
        for bboxes_for_class in bboxes.iter() {
            for b in bboxes_for_class.iter() {
                if count >= self.top_k {
                    break;
                }
                result.push(*b);
                count += 1;
            }
        }

        result
    }

    pub fn show(&self, mut image: ImageBuffer<Rgb<u8>, Vec<u8>>, bboxes: &[Bbox])
    {
        let w_ratio = 640_f64 / self.w as f64;
        let h_ratio = 640_f64 / self.h as f64;
        let green = Rgb([0u8,   255u8, 0u8]);
        for bbox in bboxes.iter() {
            let width = ((bbox.xmax - bbox.xmin) / w_ratio) as u32;
            let height = ((bbox.ymax - bbox.ymin) / h_ratio) as u32;
            let xmin = ((bbox.xmin / w_ratio) as u32).clamp(0, self.w);
            let ymin = ((bbox.ymin / h_ratio) as u32).clamp(0, self.h);
            let rect = Rect::at(xmin as i32, ymin as i32).of_size(width, height);
            imageproc::drawing::draw_hollow_rect_mut(&mut image, rect, green);
            let height = 20;
            let scale = PxScale {
                x: height as f32 * 1.5,
                y: height as f32,
            };
            let font = match FontRef::try_from_slice(include_bytes!("../font/OpenSans-Regular.ttf")) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to load font: {}", e); // 打印错误信息
                    return; // 直接返回，终止函数执行
                }
            };
            let label = bbox.name(&DROPLET_CLASS_LABELS);
            imageproc::drawing::draw_text_mut(&mut image, green, xmin as i32, ymin as i32 - height, scale, &font, &label)
        }
        image.save("./result_2.jpg").unwrap();
    }
}