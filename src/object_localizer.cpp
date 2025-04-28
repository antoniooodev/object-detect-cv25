#include "object_localizer.hpp"
#include <iostream>

std::vector<cv::Point2f> ObjectLocalizer::extractDetectedPoints(
    const std::vector<cv::KeyPoint> &keypointsModel,
    const std::vector<cv::KeyPoint> &keypointsTest,
    const std::vector<cv::DMatch> &matches)
{
    std::vector<cv::Point2f> ptsTest;
    ptsTest.reserve(matches.size());
    for (const auto &m : matches)
    {
        if (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(keypointsModel.size()) &&
            m.trainIdx >= 0 && m.trainIdx < static_cast<int>(keypointsTest.size()))
        {
            ptsTest.push_back(keypointsTest[m.trainIdx].pt);
        }
    }
    return ptsTest;
}

std::vector<cv::Point2f> ObjectLocalizer::filterPointsByDistance(
    const std::vector<cv::Point2f> &points,
    double maxDistance)
{
    if (points.empty())
        return {};

    // Compute center
    cv::Point2f center(0, 0);
    for (const auto &p : points)
        center += p;
    center *= (1.0f / points.size());

    // Filter points based on distance to center
    std::vector<cv::Point2f> filtered;
    for (const auto &p : points)
    {
        if (cv::norm(p - center) <= maxDistance)
            filtered.push_back(p);
    }

    return filtered;
}

std::vector<cv::Point2f> ObjectLocalizer::clusterMeanShift(
    const std::vector<cv::Point2f> &points,
    double bandwidth)
{
    if (points.empty())
        return {};

    // Mean Shift clustering
    std::vector<cv::Point2f> shiftedPoints = points;

    bool converged = false;
    int maxIterations = 100;
    double eps = 1e-3;

    for (int iter = 0; iter < maxIterations && !converged; ++iter)
    {
        converged = true;

        for (size_t i = 0; i < shiftedPoints.size(); ++i)
        {
            cv::Point2f mean(0, 0);
            int count = 0;

            for (const auto &p : shiftedPoints)
            {
                if (cv::norm(p - shiftedPoints[i]) < bandwidth)
                {
                    mean += p;
                    count++;
                }
            }

            if (count > 0)
            {
                mean *= (1.0f / count);
                if (cv::norm(mean - shiftedPoints[i]) > eps)
                {
                    shiftedPoints[i] = mean;
                    converged = false;
                }
            }
        }
    }

    // Take points close to the densest mode
    cv::Point2f finalCenter(0, 0);
    for (const auto &p : shiftedPoints)
        finalCenter += p;
    finalCenter *= (1.0f / shiftedPoints.size());

    std::vector<cv::Point2f> clusteredPoints;
    for (const auto &p : points)
    {
        if (cv::norm(p - finalCenter) < bandwidth * 1.1) // Slightly increased factor
            clusteredPoints.push_back(p);
    }

    return clusteredPoints;
}

void ObjectLocalizer::drawBox(
    cv::Mat &image,
    const std::vector<cv::Point2f> &points,
    const cv::Scalar &color,
    int thickness)
{
    if (points.size() < 2)
        return;

    // Compute minimum-area rotated rectangle
    cv::RotatedRect rbox = cv::minAreaRect(points);

    // Expand the box by 15% (balanced between 10% and 20%)
    rbox.size.width *= 1.15f;
    rbox.size.height *= 1.15f;

    cv::Point2f corners[4];
    rbox.points(corners);

    // Draw its four edges
    for (int i = 0; i < 4; ++i)
    {
        cv::line(image, corners[i], corners[(i + 1) % 4], color, thickness);
    }
}

cv::Rect ObjectLocalizer::getBoundingBoxFromHomography(
    const std::vector<cv::KeyPoint> &keypointsModel,
    const std::vector<cv::KeyPoint> &keypointsTest,
    const std::vector<cv::DMatch> &matches,
    const cv::Size &modelSize)
{
    if (matches.size() < 4)
        return cv::Rect();

    // Prepare source and destination points
    std::vector<cv::Point2f> srcPoints, dstPoints;
    for (const auto &match : matches)
    {
        srcPoints.push_back(keypointsModel[match.queryIdx].pt);
        dstPoints.push_back(keypointsTest[match.trainIdx].pt);
    }

    // Find homography
    cv::Mat H = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 3.0);

    if (H.empty())
        return cv::Rect();

    // Define the model's corners
    std::vector<cv::Point2f> modelCorners(4);
    modelCorners[0] = cv::Point2f(0, 0);
    modelCorners[1] = cv::Point2f(modelSize.width, 0);
    modelCorners[2] = cv::Point2f(modelSize.width, modelSize.height);
    modelCorners[3] = cv::Point2f(0, modelSize.height);

    // Transform corners
    std::vector<cv::Point2f> transformedCorners;
    cv::perspectiveTransform(modelCorners, transformedCorners, H);

    // Find bounding rectangle of transformed corners
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = 0, maxY = 0;

    for (const auto &pt : transformedCorners)
    {
        minX = std::min(minX, (int)pt.x);
        minY = std::min(minY, (int)pt.y);
        maxX = std::max(maxX, (int)pt.x);
        maxY = std::max(maxY, (int)pt.y);
    }

    // Expand the bounding box by 15% (balanced value)
    int width = maxX - minX;
    int height = maxY - minY;
    minX = std::max(0, minX - width / 7);  // ~15%
    minY = std::max(0, minY - height / 7); // ~15%
    maxX = maxX + width / 7;
    maxY = maxY + height / 7;

    return cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

// Balanced function for all objects
cv::Rect ObjectLocalizer::adaptiveBoundingBox(
    const std::vector<cv::KeyPoint> &keypointsTest,
    const std::vector<cv::DMatch> &matches,
    const std::string &objectType)
{
    // Extract test points from matches
    std::vector<cv::Point2f> testPoints;
    for (const auto &match : matches)
    {
        testPoints.push_back(keypointsTest[match.trainIdx].pt);
    }

    if (testPoints.empty())
        return cv::Rect();

    // Calculate bounding box
    cv::Rect bbox = cv::boundingRect(testPoints);

    // Balanced padding factor for all objects
    double paddingFactor = 0.18; // 18% as a balanced value

    // Apply slight variations by type
    if (objectType.find("power_drill") != std::string::npos)
    {
        paddingFactor = 0.20; // Slightly higher for power drill, but not excessive
    }
    else if (objectType.find("mustard") != std::string::npos ||
             objectType.find("sugar") != std::string::npos)
    {
        paddingFactor = 0.16; // Slightly lower for bottles and boxes
    }

    // Apply padding
    int padX = static_cast<int>(bbox.width * paddingFactor);
    int padY = static_cast<int>(bbox.height * paddingFactor);

    // Ensure coordinates are valid
    int x = std::max(0, bbox.x - padX);
    int y = std::max(0, bbox.y - padY);
    int width = bbox.width + 2 * padX;
    int height = bbox.height + 2 * padY;

    return cv::Rect(x, y, width, height);
}