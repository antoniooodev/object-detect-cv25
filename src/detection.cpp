#include "detection.hpp"

std::vector<cv::KeyPoint> Detection::detectKeypoints(const cv::Mat &image, const cv::Mat &mask)
{
    // Use SIFT with optimized parameters for industrial objects
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
        0,    // nfeatures (0 = no limit)
        3,    // nOctaveLayers
        0.04, // contrastThreshold
        10.0, // edgeThreshold
        1.6   // sigma
    );

    std::vector<cv::KeyPoint> keypoints;
    sift->detect(image, keypoints, mask);

    return keypoints;
}

cv::Mat Detection::computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Mat descriptors;
    sift->compute(image, keypoints, descriptors);
    return descriptors;
}