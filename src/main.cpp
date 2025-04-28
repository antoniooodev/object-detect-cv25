#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include "preprocessing.hpp"
#include "detection.hpp"
#include "dataloader.hpp"
#include "matching.hpp"
#include "object_localizer.hpp"

namespace fs = std::__fs::filesystem;

// Balanced parameters for all objects
struct DetectionParams
{
    int matchesThreshold;
    int minInliers;
    double clusterBandwidth;
    double maxDistanceFromCenter;
    double ransacThreshold;
};

DetectionParams getObjectParams(const std::string &objectKey)
{
    DetectionParams params;
    // Base balanced parameters
    params.matchesThreshold = 8;
    params.minInliers = 5;
    params.clusterBandwidth = 45.0;      // Intermediate value between 40 and 50
    params.maxDistanceFromCenter = 60.0; // Intermediate value
    params.ransacThreshold = 3.0;

    // Small adjustments per object type
    if (objectKey.find("power_drill") != std::string::npos)
    {
        params.matchesThreshold = 6;
        params.minInliers = 4;
        params.clusterBandwidth = 50.0;      // Reduced from previous version
        params.maxDistanceFromCenter = 70.0; // Reduced from previous version
    }
    else if (objectKey.find("mustard") != std::string::npos)
    {
        params.clusterBandwidth = 48.0;
    }

    return params;
}

int main()
{
    fs::path rootPath("../data/object_detection_dataset/");
    fs::path resultsPath("../data/results/");

    // Create results directory if it doesn't exist
    if (!fs::exists(resultsPath))
    {
        fs::create_directories(resultsPath);
    }

    // Open log file
    std::ofstream logFile(resultsPath / "detection_results.txt");
    if (!logFile.is_open())
    {
        std::cerr << "Failed to open log file" << std::endl;
        return 1;
    }

    FileSystemDataLoader loader;
    auto code = loader.checkIntegrity(rootPath);
    if (code != IntegrityCode::OK)
    {
        std::cerr << "Dataset integrity error: " << static_cast<int>(code) << std::endl;
        return static_cast<int>(code);
    }

    for (const auto &key : loader.listObjectKeys(rootPath))
    {
        std::cout << "Processing object: " << key << std::endl;
        logFile << "Processing object: " << key << std::endl;

        // Get specific parameters for this object
        DetectionParams params = getObjectParams(key);

        fs::path outDir = resultsPath / key;
        if (!fs::exists(outDir))
        {
            fs::create_directories(outDir);
        }

        // Load model views
        auto modelViews = loader.loadModelViews(rootPath, key);
        std::vector<cv::Mat> modelDescriptors;
        std::vector<std::string> modelNames;
        std::vector<std::vector<cv::KeyPoint>> modelKeypoints;
        std::vector<cv::Mat> modelMasks;

        // Process each model view
        for (const auto &mv : modelViews)
        {
            cv::Mat grayModel;
            cv::cvtColor(mv.color, grayModel, cv::COLOR_BGR2GRAY);

            // Preprocess model image
            cv::Mat processedModel = Preprocessing::reduceNoise(grayModel);

            // Detect keypoints using mask
            auto kpModel = Detection::detectKeypoints(processedModel, mv.mask);

            // Compute descriptors
            auto descModel = Detection::computeDescriptors(processedModel, kpModel);

            // Store model information
            modelDescriptors.push_back(descModel);
            modelNames.push_back(mv.name);
            modelKeypoints.push_back(kpModel);
            modelMasks.push_back(mv.mask);

            std::cout << "  Model view '" << mv.name << "' keypoints: " << kpModel.size() << std::endl;
            logFile << "  Model view '" << mv.name << "' keypoints: " << kpModel.size() << std::endl;
        }

        // Process test images
        auto testImages = loader.listTestImages(rootPath, key);
        for (const auto &ti : testImages)
        {
            std::cout << "  Processing test image: " << ti.name << std::endl;
            logFile << "  Processing test image: " << ti.name << std::endl;

            // Load test image
            cv::Mat timg = cv::imread(ti.path.string());
            if (timg.empty())
            {
                std::cerr << "  Failed to read image: " << ti.name << std::endl;
                logFile << "  Failed to read image: " << ti.name << std::endl;
                continue;
            }

            // Convert to grayscale and preprocess
            cv::Mat grayTest;
            cv::cvtColor(timg, grayTest, cv::COLOR_BGR2GRAY);
            cv::Mat processedTestImage = Preprocessing::reduceNoise(grayTest);

            // Detect features in test image
            auto kpTest = Detection::detectKeypoints(processedTestImage);
            auto descTest = Detection::computeDescriptors(processedTestImage, kpTest);

            if (descTest.empty())
            {
                std::cerr << "  Warning: No descriptors found in test image: " << ti.name << std::endl;
                logFile << "  Warning: No descriptors found in test image: " << ti.name << std::endl;
                continue;
            }

            // Find the best matching model view
            size_t bestModelIdx = 0;
            int maxGoodMatches = 0;
            std::vector<cv::DMatch> bestMatches;
            std::vector<cv::DMatch> bestInliers;

            for (size_t m = 0; m < modelDescriptors.size(); ++m)
            {
                // Match descriptors
                auto goodMatches = Matching::matchDescriptors(modelDescriptors[m], descTest);

                // Find RANSAC inliers
                auto inlierMatches = Matching::findRansacInliers(
                    modelKeypoints[m], kpTest, goodMatches, params.ransacThreshold);

                std::cout << "    Model View: " << modelNames[m]
                          << " - Good Matches: " << goodMatches.size()
                          << " - Inliers: " << inlierMatches.size() << std::endl;
                logFile << "    Model View: " << modelNames[m]
                        << " - Good Matches: " << goodMatches.size()
                        << " - Inliers: " << inlierMatches.size() << std::endl;

                // Track the best model view
                if ((int)goodMatches.size() > maxGoodMatches)
                {
                    maxGoodMatches = goodMatches.size();
                    bestModelIdx = m;
                    bestMatches = goodMatches;
                    bestInliers = inlierMatches;
                }
            }

            // If we have enough matches
            if (maxGoodMatches >= params.matchesThreshold)
            {
                bool detectionSucceeded = false;
                cv::Rect detectedBox;

                // Try different strategies in order of preference

                // 1. Try adaptive bounding box (works well for power_drill)
                if (key.find("power_drill") != std::string::npos)
                {
                    detectedBox = ObjectLocalizer::adaptiveBoundingBox(
                        kpTest, bestInliers.size() >= params.minInliers ? bestInliers : bestMatches, key);

                    if (detectedBox.width > 0 && detectedBox.height > 0)
                    {
                        detectionSucceeded = true;
                    }
                }

                // 2. Try homography (usually the best)
                if (!detectionSucceeded && bestInliers.size() >= params.minInliers)
                {
                    cv::Size modelSize(modelMasks[bestModelIdx].cols, modelMasks[bestModelIdx].rows);
                    detectedBox = ObjectLocalizer::getBoundingBoxFromHomography(
                        modelKeypoints[bestModelIdx], kpTest, bestInliers, modelSize);

                    if (detectedBox.width > 0 && detectedBox.height > 0 &&
                        detectedBox.width < 600 && detectedBox.height < 600)
                    {
                        detectionSucceeded = true;
                    }
                }

                // 3. Fallback to clustering
                if (!detectionSucceeded)
                {
                    // Use matches with the strongest confidence
                    std::vector<cv::DMatch> matchesToUse = bestInliers.size() >= 4 ? bestInliers : bestMatches;

                    std::vector<cv::Point2f> testPoints = ObjectLocalizer::extractDetectedPoints(
                        modelKeypoints[bestModelIdx], kpTest, matchesToUse);

                    // Filter and cluster points
                    auto filteredPoints = ObjectLocalizer::filterPointsByDistance(
                        testPoints, params.maxDistanceFromCenter);
                    auto clusteredPoints = ObjectLocalizer::clusterMeanShift(
                        filteredPoints, params.clusterBandwidth);

                    if (!clusteredPoints.empty())
                    {
                        // Create a separate image for rotated box
                        cv::Mat boxedImg = timg.clone();
                        ObjectLocalizer::drawBox(boxedImg, clusteredPoints, cv::Scalar(0, 255, 0), 2);

                        // Also use regular bounding box for consistency
                        detectedBox = cv::boundingRect(clusteredPoints);

                        // Expand the bounding box with a balanced value
                        int padding = std::max(5, std::min(detectedBox.width, detectedBox.height) / 6); // ~16%
                        detectedBox.x = std::max(0, detectedBox.x - padding);
                        detectedBox.y = std::max(0, detectedBox.y - padding);
                        detectedBox.width = std::min(timg.cols - detectedBox.x, detectedBox.width + 2 * padding);
                        detectedBox.height = std::min(timg.rows - detectedBox.y, detectedBox.height + 2 * padding);

                        // Draw regular bounding box on the original image
                        cv::rectangle(timg, detectedBox, cv::Scalar(0, 255, 0), 2);

                        // Save both images
                        fs::path resultPath = outDir / ("result_" + ti.name);
                        fs::path boxedPath = outDir / ("rotated_" + ti.name);
                        cv::imwrite(resultPath.string(), timg);
                        cv::imwrite(boxedPath.string(), boxedImg);

                        detectionSucceeded = true;
                    }
                }

                // If we succeeded with any of the strategies
                if (detectionSucceeded)
                {
                    // Draw regular bounding box
                    cv::rectangle(timg, detectedBox, cv::Scalar(0, 255, 0), 2);

                    // Save the result
                    fs::path resultPath = outDir / ("result_" + ti.name);
                    cv::imwrite(resultPath.string(), timg);

                    // Log detection
                    logFile << "  " << ti.name << ": Object detected at "
                            << detectedBox.x << "," << detectedBox.y << " - "
                            << detectedBox.x + detectedBox.width << ","
                            << detectedBox.y + detectedBox.height
                            << " (matches: " << maxGoodMatches
                            << ", inliers: " << bestInliers.size() << ")" << std::endl;
                }
                else
                {
                    std::cout << "  No valid bounding box found for: " << ti.name << std::endl;
                    logFile << "  " << ti.name << ": No valid bounding box found" << std::endl;
                }
            }
            else
            {
                std::cout << "  Not enough matches for image: " << ti.name << std::endl;
                logFile << "  " << ti.name << ": Not enough matches (best: " << maxGoodMatches
                        << ", inliers: " << bestInliers.size() << ")" << std::endl;
            }
        }
    }

    logFile.close();
    return 0;
}