#include "matching.hpp"

std::vector<cv::DMatch> Matching::matchDescriptors(
    const cv::Mat& modelDescriptors,
    const cv::Mat& testDescriptors,
    float nndrRatio)
{
    std::vector<std::vector<cv::DMatch>> bfMatches;

    // Create brute-force matcher; SIFT requires L2 norm
    cv::BFMatcher matcher(cv::NORM_L2);

    // Store two best matches for each descriptor in the model view
    matcher.knnMatch(modelDescriptors, testDescriptors, bfMatches, 2);

    // Filter matches with NNDR test
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : bfMatches)
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
