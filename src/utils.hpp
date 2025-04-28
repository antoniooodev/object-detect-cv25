// utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>

// Holds the predicted bounding box
struct Prediction {
    std::string objectName;
    cv::Rect bbox;
};

// Holds the correct bounding box
struct GroundTruth {
    std::string objectName;
    cv::Rect bbox;
};

// Computes Intersection over Union for two boxes
float computeIoU(const cv::Rect& boxA, const cv::Rect& boxB);

// Reads all ground truths for a certain object 
std::vector<GroundTruth> loadAllGroundTruths(const std::filesystem::path& rootPath, const std::string& objectKey);

// Applies IoU threshold and outputs performance metrics
void evaluatePerformance(
    const std::vector<Prediction>& predictions,
    const std::vector<GroundTruth>& groundTruths,
    const std::string& objectKey
);

#endif