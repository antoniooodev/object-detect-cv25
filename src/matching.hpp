#ifndef MATCHING_HPP
#define MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class Matching
{
public:
    // Match descriptors with NNDR test
    static std::vector<cv::DMatch> matchDescriptors(
        const cv::Mat &modelDescriptors,
        const cv::Mat &testDescriptors,
        float nndrRatio = 0.75f);

    // Find object in test image
    static bool findObject(
        const std::vector<cv::Mat> &modelDescriptors,
        const std::vector<std::string> &modelNames,
        const cv::Mat &testDescriptors,
        int matchesThreshold,
        std::string &bestModel,
        int &maxGoodMatches);

    // Return top-N model matches
    static std::vector<std::pair<int, std::vector<cv::DMatch>>> matchTopNModels(
        const std::vector<cv::Mat> &modelDescriptors,
        const cv::Mat &testDescriptors,
        int N);

    // Find geometric inliers using RANSAC
    static std::vector<cv::DMatch> findRansacInliers(
        const std::vector<cv::KeyPoint> &keypointsModel,
        const std::vector<cv::KeyPoint> &keypointsTest,
        const std::vector<cv::DMatch> &matches,
        double ransacThreshold = 3.0);
};

#endif // MATCHING_HPP