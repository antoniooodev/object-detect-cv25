#include <opencv2/opencv.hpp>
#include <filesystem>
#include "preprocessing.hpp"
#include "detection.hpp"
#include "dataloader.hpp"
#include "matching.hpp" 

namespace fs = std::filesystem;

//namespace fs = std::__fs::filesystem;

int main()
{
    fs::path rootPath("../data/object_detection_dataset/");

    FileSystemDataLoader loader;
    auto code = loader.checkIntegrity(rootPath);
    if (code != IntegrityCode::OK) {
        std::cerr << "Dataset integrity error: " << static_cast<int>(code) << std::endl;
        return static_cast<int>(code);
    }

    for (const auto &key : loader.listObjectKeys(rootPath)) {
        std::cout << "Processing object: " << key << std::endl;

        // per-object results folder
        fs::path outDir = fs::path("../data/results") / key;
        if (!fs::exists(outDir)) {
            fs::create_directories(outDir);
        }

        // 1) Extract & store all model‐view descriptors
        auto modelViews = loader.loadModelViews(rootPath, key);
        std::vector<cv::Mat> modelDescriptors;
        std::vector<std::string> modelNames;
        for (const auto &mv : modelViews) {
            cv::Mat grayModel;
            cv::cvtColor(mv.color, grayModel, cv::COLOR_BGR2GRAY);
            auto kpModel = Detection::detectKeypoints(grayModel, mv.mask);
            auto descModel = Detection::computeDescriptors(grayModel, kpModel);
            modelDescriptors.push_back(descModel);
            modelNames.push_back(mv.name);
            std::cout << "  Model view '" << mv.name << "' keypoints: " << kpModel.size() << std::endl;
        }


        // 2) Test images (no masks)
        auto testImages = loader.listTestImages(rootPath, key);
        for (const auto &ti : testImages) {
            cv::Mat timg = cv::imread(ti.path.string());
            if (timg.empty()) {
                std::cerr << "  Failed to read image: " << ti.name << std::endl;
                continue;
            }

            cv::Mat graytimg;
            cv::cvtColor(timg, graytimg, cv::COLOR_BGR2GRAY);
            cv::Mat processedTestImage = Preprocessing::reduceNoise(graytimg);

            auto kpTest   = Detection::detectKeypoints(processedTestImage);
            auto descTest = Detection::computeDescriptors(processedTestImage, kpTest);
            // Output the results
            std::cout << "  Test image: " << ti.name << " - Keypoints: " << kpTest.size() << std::endl;

            // Object detection threshold
            int matchesThreshold = 10;
            bool objectPresence = false;
            int maxGoodMatches = 0;
            std::string bestModel;

            // Match against every model view
            for (size_t m = 0; m < modelDescriptors.size(); ++m) {
                auto goodMatches = Matching::matchDescriptors(modelDescriptors[m], descTest);
                std::cout << "    Model View: " << modelNames[m]
                          << "  Good Matches: " << goodMatches.size() << std::endl;

                // Track best model view with the most good matches
                if (goodMatches.size() > maxGoodMatches) {
                    maxGoodMatches = goodMatches.size();
                    bestModel = modelNames [m];
                }
            }

            // If the best view had more good matches than the threshold, consider the object found
            if (maxGoodMatches > matchesThreshold) {
                objectPresence = true;
                std::cout << "Object '" << key << "' detected in test image: " << ti.name
                          << " (Best model: " << bestModel
                          << " with " << maxGoodMatches << " matches)" << std::endl;

            }
            
            // TODO
            // If object was found: draw matching box
            if (objectPresence) {
                cv::Mat outputImage;
                cv::drawKeypoints(
                    processedTestImage,
                    kpTest,
                    outputImage,
                    cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
                );

                fs::path savePath = outDir / ("sift_" + ti.name);
                cv::imwrite(savePath.string(), outputImage);
            }
        }
    }

    return 0;
}
