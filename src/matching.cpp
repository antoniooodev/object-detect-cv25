#include "matching.hpp"

std::vector<cv::DMatch> Matching::matchDescriptors(
    const cv::Mat &modelDescriptors,
    const cv::Mat &testDescriptors,
    float nndrRatio)
{
    std::vector<std::vector<cv::DMatch>> bfMatches;

    // Create brute-force matcher; SIFT requires L2 norm
    cv::BFMatcher matcher(cv::NORM_L2);

    // Store two best matches for each descriptor in the model view
    matcher.knnMatch(modelDescriptors, testDescriptors, bfMatches, 2);

    // Filter matches with NNDR test
    std::vector<cv::DMatch> goodMatches;
    for (const auto &match : bfMatches)
    {
        // Discard entries with less than 2 matches
        if (match.size() >= 2)
        {
            // Check if the best match is significantly better than the second best
            if (match[0].distance < nndrRatio * match[1].distance)
            {
                // Save the best match
                goodMatches.push_back(match[0]);
            }
        }
    }

    // Return best matches found
    return goodMatches;
}

bool Matching::findObject(
    const std::vector<cv::Mat> &modelDescriptors,
    const std::vector<std::string> &modelNames,
    const cv::Mat &testDescriptors,
    int matchesThreshold,
    std::string &bestModel,
    int &maxGoodMatches)
{
    maxGoodMatches = 0;

    // Match against every model view
    for (int m = 0; m < modelDescriptors.size(); m++)
    {
        auto goodMatches = Matching::matchDescriptors(modelDescriptors[m], testDescriptors);
        std::cout << "    Model View: " << modelNames[m]
                  << "  Good Matches: " << goodMatches.size() << std::endl;

        // Track best model view with the most good matches
        if (goodMatches.size() > maxGoodMatches)
        {
            maxGoodMatches = goodMatches.size();
            bestModel = modelNames[m];
        }
    }

    // If the best view had more good matches than the threshold, consider the object found
    return (maxGoodMatches > matchesThreshold);
}

std::vector<std::pair<int, std::vector<cv::DMatch>>> Matching::matchTopNModels(
    const std::vector<cv::Mat> &modelDescriptors,
    const cv::Mat &testDescriptors,
    int N)
{
    std::vector<std::pair<int, std::vector<cv::DMatch>>> allMatches;
    allMatches.reserve(modelDescriptors.size());

    for (int i = 0; i < static_cast<int>(modelDescriptors.size()); ++i)
    {
        auto gm = matchDescriptors(modelDescriptors[i], testDescriptors);
        allMatches.emplace_back(i, std::move(gm));
    }

    std::sort(allMatches.begin(), allMatches.end(),
              [](auto &a, auto &b)
              {
                  return a.second.size() > b.second.size();
              });

    if (static_cast<int>(allMatches.size()) > N)
        allMatches.resize(N);

    return allMatches;
}

std::vector<cv::DMatch> Matching::findRansacInliers(
    const std::vector<cv::KeyPoint> &keypointsModel,
    const std::vector<cv::KeyPoint> &keypointsTest,
    const std::vector<cv::DMatch> &matches,
    double ransacThreshold)
{
    // Not enough matches for RANSAC
    if (matches.size() < 4)
        return matches;

    std::vector<cv::Point2f> ptsModel, ptsTest;
    for (const auto &match : matches)
    {
        ptsModel.push_back(keypointsModel[match.queryIdx].pt);
        ptsTest.push_back(keypointsTest[match.trainIdx].pt);
    }

    // Calculate homography with RANSAC
    std::vector<uchar> inliersMask;
    cv::Mat H = cv::findHomography(ptsModel, ptsTest, cv::RANSAC, ransacThreshold, inliersMask);

    // If homography couldn't be computed, return all matches
    if (H.empty())
        return matches;

    // Filter matches using inliers mask
    std::vector<cv::DMatch> inlierMatches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (inliersMask[i])
        {
            inlierMatches.push_back(matches[i]);
        }
    }

    return inlierMatches;
}