// object_localizer.cpp
#include "object_localizer.hpp"
#include <iostream>

//Given matched keypoints between model and test images, returns only the test-image point coordinates.
std::vector<cv::Point2f> ObjectLocalizer::extractDetectedPoints(
    const std::vector<cv::KeyPoint>& keypointsModel,
    const std::vector<cv::KeyPoint>& keypointsTest,
    const std::vector<cv::DMatch>& matches)
{
    std::vector<cv::Point2f> ptsTest;
    ptsTest.reserve(matches.size());
    for (const auto& m : matches) {
        if (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(keypointsModel.size()) &&
            m.trainIdx >= 0 && m.trainIdx < static_cast<int>(keypointsTest.size()))
        {
            ptsTest.push_back(keypointsTest[m.trainIdx].pt);
        }
    }
    return ptsTest;
}
// Performs mean-shift clustering on the input 2D points.
// Returns the points belonging to the largest-density cluster.
// Parameters:
//   bandwidth – radius within which neighbors influence shift
//   eps       – convergence threshold for mode updates
//   maxIter   – maximum iterations per point
std::vector<cv::Point2f> ObjectLocalizer::clusterMeanShift(
    const std::vector<cv::Point2f>& points,
    float bandwidth,
    float eps,
    int maxIter)
{
    size_t n = points.size();
    std::vector<cv::Point2f> modes(n);
    // 1) find mode for each point
    for (size_t i = 0; i < n; ++i) {
        cv::Point2f mode = points[i];
        for (int it = 0; it < maxIter; ++it) {
            cv::Point2f sum(0,0);
            int count = 0;
            for (auto &p : points) {
                float d = cv::norm(p - mode);
                if (d <= bandwidth) {
                    sum += p;
                    ++count;
                }
            }
            if (count == 0) break;
            cv::Point2f newMode = sum * (1.f / count);
            if (cv::norm(newMode - mode) < eps)
                break;
            mode = newMode;
        }
        modes[i] = mode;
    }

    // 2) cluster labels by mode proximity
    std::vector<int> labels(n, -1);
    int nextLabel = 0;
    for (size_t i = 0; i < n; ++i) {
        if (labels[i] != -1) continue;
        labels[i] = nextLabel;
        for (size_t j = i + 1; j < n; ++j) {
            if (labels[j] == -1 && cv::norm(modes[i] - modes[j]) <= bandwidth) {
                labels[j] = nextLabel;
            }
        }
        ++nextLabel;
    }

    // 3) find largest cluster
    std::vector<int> counts(nextLabel, 0);
    for (auto L : labels) if (L >= 0) ++counts[L];
    int bestLabel = std::max_element(counts.begin(), counts.end()) - counts.begin();

    // 4) collect cluster points
    std::vector<cv::Point2f> clusterPts;
    for (size_t i = 0; i < n; ++i) {
        if (labels[i] == bestLabel)
            clusterPts.push_back(points[i]);
    }
    return clusterPts;
}

// Computes the minimum-area rotated rectangle over the given points and draws its four edges onto 'image'.
cv::RotatedRect ObjectLocalizer::drawBox(cv::Mat& image,
    const std::vector<cv::Point2f>& points,
    const cv::Scalar& color,
    int thickness)
{
if (points.size() < 2)
    return cv::RotatedRect();

cv::RotatedRect rbox = cv::minAreaRect(points);
cv::Point2f corners[4];
rbox.points(corners);
for (int i = 0; i < 4; ++i) {
    cv::line(image, corners[i], corners[(i + 1) % 4], color, thickness);
}
return rbox;
}