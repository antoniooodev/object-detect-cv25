#ifndef OBJECT_LOCALIZER_HPP
#define OBJECT_LOCALIZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ObjectLocalizer
{
public:
    // Extract detected points from matches
    static std::vector<cv::Point2f> extractDetectedPoints(
        const std::vector<cv::KeyPoint> &keypointsModel,
        const std::vector<cv::KeyPoint> &keypointsTest,
        const std::vector<cv::DMatch> &matches);

    // Filter points based on distance from center
    static std::vector<cv::Point2f> filterPointsByDistance(
        const std::vector<cv::Point2f> &points,
        double maxDistance);

    // Perform MeanShift clustering on points
    static std::vector<cv::Point2f> clusterMeanShift(
        const std::vector<cv::Point2f> &points,
        double bandwidth);

    // Draw bounding box around points
    static void drawBox(
        cv::Mat &image,
        const std::vector<cv::Point2f> &points,
        const cv::Scalar &color = cv::Scalar(0, 255, 0),
        int thickness = 2);

    // Get bounding box using homography
    static cv::Rect getBoundingBoxFromHomography(
        const std::vector<cv::KeyPoint> &keypointsModel,
        const std::vector<cv::KeyPoint> &keypointsTest,
        const std::vector<cv::DMatch> &matches,
        const cv::Size &modelSize);

    // Get adaptive bounding box with type-specific padding
    static cv::Rect adaptiveBoundingBox(
        const std::vector<cv::KeyPoint> &keypointsTest,
        const std::vector<cv::DMatch> &matches,
        const std::string &objectType);
};

#endif // OBJECT_LOCALIZER_HPP