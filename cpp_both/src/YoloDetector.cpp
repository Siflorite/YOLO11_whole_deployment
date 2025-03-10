#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "YoloDetector.h"
#include "MiscFunctions.h"
#include "omp.h"

YoloDetector::YoloDetector(ModelType type, const std::string& model_path, 
    float conf_thres, float iou_thres)
    : m_type(type), m_conf_thres(conf_thres), m_iou_thres(iou_thres) {
    
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3);
    // 初始化模型
    if (type == ONNX) {
        init_onnx_model(model_path);
    } else {
        init_torchscript_model(model_path);
    }
}

std::vector<Detection> YoloDetector::predict(cv::Mat& image) {
    if (m_type == ONNX) {
        return predict_onnx(image);
    } else {
        return predict_torchscript(image);
    }
}

void YoloDetector::init_onnx_model(const std::string& path) {
    Ort::SessionOptions options = Ort::SessionOptions();
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();

    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
    std::vector<const char*> option_keys = {
        "device_id",
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_fp16_enable"
    };

    std::vector<const char*> option_values = {
        "0",
        "1",
        "./cache",
        "1"
    };
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options,
        option_keys.data(), option_values.data(), option_keys.size()));

    // 定义优先级顺序
    std::vector<std::pair<std::string, std::function<void()>>> providers = {
        {"TensorrtExecutionProvider", [&]() {
            options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
            std::cout << "ONNX 使用 TensorRT 加速" << std::endl;
        }},
        {"CUDAExecutionProvider", [&]() {
            OrtCUDAProviderOptions cudaOption;
            options.AppendExecutionProvider_CUDA(cudaOption);
            std::cout << "ONNX 使用 CUDA 加速" << std::endl;
        }}
    };

    // 按优先级顺序尝试启用提供者
    bool providerEnabled = false;
    for (const auto& [providerName, enableProvider] : providers) {
        if (std::find(availableProviders.begin(), availableProviders.end(), providerName) != availableProviders.end()) {
            enableProvider();
            providerEnabled = true;
            break; // 启用一个提供者后退出
        }
    }

    // 如果没有启用任何提供者，则使用 CPU
    if (!providerEnabled) {
        std::cout << "ONNX 使用 CPU 推理" << std::endl;
    }

#ifdef _WIN32
    wchar_t* buffer = string_to_wchart(path);
    m_onnx_session = Ort::Session(m_onnx_env, buffer, options);
#else
    m_onnx_session = Ort::Session(m_onnx_env, path.c_str(), options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = m_onnx_session.GetInputNameAllocated(0, allocator);
    this->input_names.push_back(std::string(input_name.get()));
    this->input_names_ptrs.push_back(this->input_names[0].c_str());

    auto output_name = m_onnx_session.GetOutputNameAllocated(0, allocator);
    this->output_names.push_back(std::string(output_name.get()));
    this->output_names_ptrs.push_back(this->output_names[0].c_str());
}

void YoloDetector::init_torchscript_model(const std::string& path) {
    m_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    try {
        m_ts_module = torch::jit::load(path, m_device);
        m_ts_module.eval();
    } catch (const c10::Error& e) {
        std::cout << "Error:" << std::endl;
        std::cout << e.msg() << std::endl;
    }
}

std::vector<Detection> YoloDetector::predict_onnx(cv::Mat& image) {
    // 预处理
    auto start = std::chrono::system_clock::now();
    cv::Mat processed;
    float scale = letterbox(image, processed, {640, 640});
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    // 将HWC转换为CHW

    // cv::Mat blob = cv::dnn::blobFromImage(
    //     processed, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    // 手动blob

    std::vector<float> input_data(1 * 3 * 640 * 640);  // 提前分配
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 640; ++h) {
            for (int w = 0; w < 640; ++w) {
                // 直接访问连续内存
                const int src_idx = h * 640 * 3 + w * 3 + c;
                const int dst_idx = c * 640 * 640 + h * 640 + w;
                input_data[dst_idx] = processed.data[src_idx] * (1.0f / 255.0f);  // 合并归一化
            }
        }
    }

    // 准备输入Tensor
    std::array<int64_t, 4> input_shape{1, 3, 640, 640};
    // std::vector<float> input_data(blob.ptr<float>(), blob.ptr<float>() + 3 * 640 * 640);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_shape.data(), input_shape.size()));

    auto end = std::chrono::system_clock::now();
    std::cout << "Pre-process Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms, ";

    // 运行推理
    start = std::chrono::system_clock::now();
    auto outputs = m_onnx_session.Run(
        Ort::RunOptions{nullptr},
        this->input_names_ptrs.data(), 
        input_tensors.data(), 
        1,
        this->output_names_ptrs.data(),
        1
    );
    end = std::chrono::system_clock::now();
    std::cout << "Inference Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms, ";

    // 后处理
    float* output = outputs[0].GetTensorMutableData<float>();
    // ... [后处理逻辑，生成Detection对象] ...

    return postprocess(output, scale, image.size());
}

std::vector<Detection> YoloDetector::predict_torchscript(cv::Mat& image) {
    // 预处理
    auto start = std::chrono::system_clock::now();

    cv::Mat processed;
    float scale = letterbox(image, processed, {640, 640});
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 转换为Tensor
    torch::Tensor tensor = torch::from_blob(processed.data, 
        {processed.rows, processed.cols, 3}, torch::kByte)
        .permute({2, 0, 1})
        .unsqueeze(0)
        .to(m_device, torch::kFloat32)
        .div(255);
    
    auto end = std::chrono::system_clock::now();
    std::cout << "Pre-process Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms, ";

    // 运行推理
    start = std::chrono::system_clock::now();
    auto output = m_ts_module.forward({tensor}).toTensor()
                  .cpu().contiguous();
    end = std::chrono::system_clock::now();
    std::cout << "Inference Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms, ";

    // 后处理
    return postprocess(output.data_ptr<float>(), scale, image.size());
}

std::vector<Detection> YoloDetector::postprocess(float* output, float scale, const cv::Size& orig_size) {
    // 统一后处理逻辑
    auto start = std::chrono::system_clock::now();
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;
    std::vector<Detection> results;

    // [1*(4+4)*8400 -> 1*8400*(4+4)]
    // Requires further mending for automatic class num
    cv::Mat output0 = cv::Mat(cv::Size(8400, 8), CV_32F, output).t();
    float* output0_ptr = output0.ptr<float>();
    for (int i = 0; i < 8400; i++) {
        std::vector<float> it(output0_ptr + i * 8, output0_ptr + (i + 1) * 8);
        float conf;
        int class_id;
        auto index = std::max_element(it.begin() + 4, it.begin() + 8);
        class_id = index - it.begin() - 4;
        conf = *index;
        if (conf < this->m_conf_thres) continue;

        int centerX = (int)(it[0]);
        int centerY = (int)(it[1]);
        int width = (int)(it[2]);
        int height = (int)(it[3]);
        int left = centerX - width / 2;
        int top = centerY - height / 2;
        boxes.emplace_back(left, top, width, height);
        confs.emplace_back(conf);
        class_ids.emplace_back(class_id);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, this->m_conf_thres, this->m_iou_thres, indices);

    for (int idx : indices) {
        Detection res;
        res.box = scale_box(boxes[idx], orig_size, scale);
        res.conf = confs[idx];
        res.class_id = class_ids[idx];
        results.emplace_back(res);
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "Post-process Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms." << std::endl;
    return results;
}
