#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class Detection
{
public:
    // Function to detect keypoints using SIFT
    static std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &image);

    // Function to compute SIFT descriptors for the given keypoints
    static cv::Mat computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints);
};

#endif