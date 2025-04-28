#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Box {
    std::string label;
    cv::Rect bbox;
};

namespace Utils {
    std::vector<Box> loadGroundTruthBoxes(const std::string& filename);
    float computeIoU(const cv::Rect& boxA, const cv::Rect& boxB);
}

#endif