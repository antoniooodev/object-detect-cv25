#ifndef MATCHING_HPP  // Include guard to prevent multiple inclusions
#define MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

// This class compares descriptors between one model view and the current test image
// Brute force matching + NDDR test filtering

class Matching
{
public:
    // Static function to match descriptors with NNDR test
    static std::vector<cv::DMatch> matchDescriptors(
        const cv::Mat& modelDescriptors,
        const cv::Mat& testDescriptors,
        float nndrRatio = 0.75f);

    // Static function to match against model views and return the detection result
    static bool findObject(
        const std::vector<cv::Mat>& modelDescriptors,
        const std::vector<std::string>& modelNames,
        const cv::Mat& testDescriptors,
        int matchesThreshold,
        std::string& bestModel,
        int& maxGoodMatches);
};

#endif // MATCHING_HPP
