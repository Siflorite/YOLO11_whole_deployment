#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

enum ModelType { ONNX, TORCHSCRIPT };

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

class YoloDetector {
public:
    YoloDetector(ModelType type, const std::string& model_path, 
        float conf_thres = 0.4, float iou_thres = 0.5);
    std::vector<Detection> predict(cv::Mat& image);

private:
    ModelType m_type;
    float m_conf_thres;
    float m_iou_thres;
    
    // TorchScript related
    torch::Device m_device{torch::kCPU};
    torch::jit::script::Module m_ts_module;

    // Onnx related
    Ort::Env m_onnx_env{ORT_LOGGING_LEVEL_WARNING, "YOLO"};
    Ort::Session m_onnx_session{nullptr};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char *> input_names_ptrs;
    std::vector<const char *> output_names_ptrs;

    void init_onnx_model(const std::string& path);
    void init_torchscript_model(const std::string& path);

    std::vector<Detection> predict_onnx(cv::Mat& image);
    std::vector<Detection> predict_torchscript(cv::Mat& image);

    std::vector<Detection> postprocess(float* output, float scale, const cv::Size& orig_size);
};
