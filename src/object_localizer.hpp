// object_localizer.hpp
#ifndef OBJECT_LOCALIZER_HPP
#define OBJECT_LOCALIZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ObjectLocalizer {
public:
    // Extract detected points in test-image from matches
    static std::vector<cv::Point2f> extractDetectedPoints(
        const std::vector<cv::KeyPoint>& keypointsModel,
        const std::vector<cv::KeyPoint>& keypointsTest,
        const std::vector<cv::DMatch>& matches);

    // mean-shift clustering of points
    static std::vector<cv::Point2f> clusterMeanShift(
        const std::vector<cv::Point2f>& points,
        float bandwidth = 80.f,
        float eps = 1.0f,
        int maxIter = 100);    

    // draw minimum-area bounding box over points
    static void drawBox(
        cv::Mat& image,
        const std::vector<cv::Point2f>& points,
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        int thickness = 2);
};

#endif // OBJECT_LOCALIZER_HPP
