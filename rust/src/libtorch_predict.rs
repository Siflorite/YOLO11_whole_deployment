use std::{path::Path, time::Instant};

use image::{Rgb, ImageBuffer};
use imageproc::rect::Rect;
use ab_glyph::{FontRef, PxScale};
use tch::{nn::ModuleT, Device, IndexOp, Tensor, TchError};

use crate::bbox::{Bbox, iou};
use crate::utils::DROPLET_CLASS_LABELS;

pub struct YOLOv8 {
    model: tch::CModule,
    pub device: tch::Device,
    h: i64,
    w: i64,
    conf_threshold: f64,
    iou_threshold: f64,
    top_k: i64,
}

impl YOLOv8 {
    pub fn new(
        weights: &Path,
        h: i64,
        w: i64,
        conf_threshold: f64,
        iou_threshold: f64,
        top_k: i64,
        device: Device,
    ) -> Result<YOLOv8,TchError> {
        let mut model = tch::CModule::load_on_device(weights, device)?;
        model.set_eval();
        model.to(device, tch::Kind::Half, true);
        Ok(
            YOLOv8 {
                model,
                device,
                h,
                w,
                conf_threshold,
                iou_threshold,
                top_k,
            }
        )
        
    }

    pub fn preprocess(&mut self, image_path: &str) -> Result<Tensor,TchError> {
        let origin_image = tch::vision::image::load(image_path)?;
        let (_, ori_h, ori_w) = origin_image.size3()?;
        self.w = ori_w;
        self.h = ori_h;
        let img = tch::vision::image::resize(&origin_image, 640, 640)?
            .unsqueeze(0)
            .to_kind(tch::Kind::Half)
            / 255.;
        let img = img.to_device(self.device);
        Ok(img)
    }

    pub fn predict(&mut self, image: &tch::Tensor) -> Vec<Bbox> {
        let start_time = Instant::now();
        let pred = self.model.forward_t(image, false);
        let elapsed_time = start_time.elapsed();
        println!("libtorch inference time:{} ms", elapsed_time.as_millis());

        let start_time = Instant::now();
        let result = self.non_max_suppression(&pred);
        let elapsed_time = start_time.elapsed();
        println!("libtorch nms time:{} ms", elapsed_time.as_millis());
        result
    }

    fn non_max_suppression(&self, pred: &tch::Tensor) -> Vec<Bbox> {
        let pred = &pred.transpose(2, 1).squeeze();
        let (npreds, pred_size) = pred.size2().unwrap();
        let nclasses = pred_size - 4;
        let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();

        let class_index = pred.i((.., 4..pred_size));
        let (pred_conf, class_label) = class_index.max_dim(-1, false);
        // pred_conf.save("pred_conf.pt").expect("pred_conf save err");
        // class_label.save("class_label.pt").expect("class_labe; save err");
        for index in 0..npreds {
            let pred = Vec::<f64>::try_from(pred.get(index)).unwrap();
            let conf = pred_conf.double_value(&[index]);
            if conf > self.conf_threshold {
                let label = class_label.int64_value(&[index]);
                if pred[(4 + label) as usize] > 0. {
                    let bbox = Bbox {
                        xmin: pred[0] - pred[2] / 2.,
                        ymin: pred[1] - pred[3] / 2.,
                        xmax: pred[0] + pred[2] / 2.,
                        ymax: pred[1] + pred[3] / 2.,
                        confidence: conf,
                        cls_index: label
                    };
                    bboxes[label as usize].push(bbox);
                }
            }
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
            let xmin = ((bbox.xmin / w_ratio) as i64).clamp(0, self.w);
            let ymin = ((bbox.ymin / h_ratio) as i64).clamp(0, self.h);
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
        image.save("./result_1.jpg").unwrap();
    }
}