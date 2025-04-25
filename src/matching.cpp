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


bool Matching::findObject(
    const std::vector<cv::Mat>& modelDescriptors,
    const std::vector<std::string>& modelNames,
    const cv::Mat& testDescriptors,
    int matchesThreshold,
    std::string& bestModel,
    int& maxGoodMatches) 
{
    maxGoodMatches = 0;

    // Match against every model view
    for (int m = 0; m < modelDescriptors.size(); m++) {
        auto goodMatches = Matching::matchDescriptors(modelDescriptors[m], testDescriptors);
        std::cout << "    Model View: " << modelNames[m]
                  << "  Good Matches: " << goodMatches.size() << std::endl;

        // Track best model view with the most good matches
        if (goodMatches.size() > maxGoodMatches) {
            maxGoodMatches = goodMatches.size();
            bestModel = modelNames[m];
        }
    }

        // If the best view had more good matches than the threshold, consider the object found
        return (maxGoodMatches > matchesThreshold);
};