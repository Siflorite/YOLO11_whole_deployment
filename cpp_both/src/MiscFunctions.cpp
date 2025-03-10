#include "MiscFunctions.h"
#include <numeric>
#include <Windows.h>

float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = (std::min)(ratio_h, ratio_w);
    return resize_scale;
}


float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_NEAREST); // Faster than INTER_AREA

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));

    return resize_scale;
}

std::vector<int> nms(const std::vector<std::vector<float>>& boxes,
    const std::vector<float>& scores, float thresh) {
    std::vector<int> indices(scores.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int curr = indices[0];
        keep.push_back(curr);
        
        std::vector<int> suppressed;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            float xx1 = max(boxes[curr][0], boxes[idx][0]);
            float yy1 = max(boxes[curr][1], boxes[idx][1]);
            float xx2 = min(boxes[curr][2], boxes[idx][2]);
            float yy2 = min(boxes[curr][3], boxes[idx][3]);
            
            float w = max(0.0f, xx2 - xx1);
            float h = max(0.0f, yy2 - yy1);
            float inter = w * h;
            float iou = inter / ((boxes[curr][2]-boxes[curr][0])*(boxes[curr][3]-boxes[curr][1]) + 
                       (boxes[idx][2]-boxes[idx][0])*(boxes[idx][3]-boxes[idx][1]) - inter);
            
            if (iou <= thresh) 
                suppressed.push_back(i);
        }
        
        std::vector<int> new_indices;
        for (size_t i = 0; i < suppressed.size(); ++i) 
            new_indices.push_back(indices[suppressed[i]]);
        indices.swap(new_indices);
    }
    return keep;
}

wchar_t* string_to_wchart(const std::string str)
{
    int len = MultiByteToWideChar(
        CP_UTF8,            // 源字符串编码（UTF-8）
        0,                  // 标志（通常为0）
        str.c_str(),        // 源字符串
        -1,                 // 自动计算长度（含终止符）
        nullptr,            // 目标缓冲区（设为nullptr获取长度）
        0
    );
    if (len == 0) {
        std::cout << "字符串长度为0！";
        return nullptr;
    }
    wchar_t* buffer = new wchar_t[len];
    // 实际转换
    MultiByteToWideChar(
        CP_UTF8, 0, str.c_str(), -1, buffer, len
    );
    return buffer;
}

cv::Rect scale_box(const cv::Rect box, const cv::Size &original, const float scale)
{
    int pad[2] = { (int)((640.0f - (float)original.width * scale) / 2.0f), 
        (int)((640.0f - (float)original.height * scale) / 2.0f) };

    int new_x = (int)std::round((float)(box.x - pad[0]) / scale);
    new_x = std::clamp(new_x, 0, original.width);
    int new_y = (int)std::round((float)(box.y - pad[1]) / scale);
    new_y = std::clamp(new_y, 0, original.height);
    int width = (int)std::round((float)box.width / scale);
    width = std::clamp(width, 0, original.width - new_x);
    int height = (int)std::round((float)box.height / scale);
    height = std::clamp(height, 0, original.height - new_y);
    return cv::Rect(new_x, new_y, width, height);
}
