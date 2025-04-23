#include "detection.hpp"

std::vector<cv::KeyPoint> Detection::detectKeypoints(const cv::Mat &image)
{
    // Create the SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints;
    // Detect keypoints in the image
    sift->detect(image, keypoints);

    return keypoints;
}

cv::Mat Detection::computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
{
    // Create the SIFT descriptor
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    cv::Mat descriptors;
    // Compute descriptors for the detected keypoints
    sift->compute(image, keypoints, descriptors);

    return descriptors;
}