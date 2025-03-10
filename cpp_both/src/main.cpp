#include "YoloDetector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

std::vector<std::string> droplet_classes {"0", "1", "2", "3"};

void show_results(std::vector<Detection> boxes, cv::Mat& image, bool bDraw) {
    for (int i = 0; i < boxes.size(); i++) {
        int x1 = boxes[i].box.x;
        int y1 = boxes[i].box.y;
        int x2 = boxes[i].box.x + boxes[i].box.width;
        int y2 = boxes[i].box.y + boxes[i].box.height;
        float conf = boxes[i].conf;
        int cls = boxes[i].class_id;
        // std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " 
        //    << conf << "  Class: " << droplet_classes[cls] << std::endl;
        
        // Draw if bDRAW=true
        if (bDraw)
        {
            std::string class_string = droplet_classes[cls] + ' ' + std::to_string(conf).substr(0, 4);
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);
            cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect text_rect(x1, y1 - 40, text_size.width + 10, text_size.height + 20);
            rectangle(image, text_rect, cv::Scalar(0, 255, 0), cv::FILLED);
            putText(image, class_string, cv::Point(x1 + 5, y1 - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            cv::imshow("prediction", image);
            cv::waitKey(1);
        }
    }
}

int main() {
    // Caution: Input size, output size and number of classes is hard-coded in YoloDetector.cpp, and number of classes is 4 in this case
    // If your model uses a different input/output size or number of classes, change them thoroughly or make it general.
    // 初始化检测器
    std::string model_path = "F:/YOLO/yolo11/rust/yolo_self/weights/yolo11n.torchscript";
    YoloDetector detector(TORCHSCRIPT, model_path, 0.35, 0.45);
    
    std::string VideoPath = "F:/YOLO/multicell/multicell1.avi";
    cv::VideoCapture cap(VideoPath);
    int width = cap.get(cv:: CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter output_video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
    int i = 0;

    while (1)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        std::cout << "Frame [" << i << "]:";
        auto detections = detector.predict(frame);
        show_results(detections, frame, true);
        output_video.write(frame);
        i++;
    }
    cv::destroyAllWindows();
    cap.release();
    output_video.release();
    return 0;
}
