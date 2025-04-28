#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class Detection
{
public:
    // Detect keypoints (with or without mask)
    static std::vector<cv::KeyPoint> detectKeypoints(
        const cv::Mat &image,
        const cv::Mat &mask = cv::Mat());

    // Compute SIFT descriptors
    static cv::Mat computeDescriptors(
        const cv::Mat &image,
        std::vector<cv::KeyPoint> &keypoints);
};

#endif // DETECTION_HPP