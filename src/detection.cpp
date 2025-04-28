//author: Antonio Tangaro

#include "detection.hpp"

std::vector<cv::KeyPoint> Detection::detectKeypoints(const cv::Mat &image)
{
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image;
    // Create the SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints;
    // Detect keypoints in the image
    sift->detect(gray, keypoints);

    return keypoints;
}
std::vector<cv::KeyPoint> Detection::detectKeypoints(const cv::Mat &img, const cv::Mat &mask) {
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpts;
    sift->detect(gray, kpts, mask);
    return kpts;
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

