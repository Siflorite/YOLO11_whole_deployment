#pragma once
#include <opencv2/opencv.hpp>

float generate_scale(cv::Mat& image, const std::vector<int>& target_size);
float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);
std::vector<int> nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores, float thresh);
wchar_t* string_to_wchart(const std::string str);
cv::Rect scale_box(const cv::Rect box, const cv::Size &original, const float scale);
