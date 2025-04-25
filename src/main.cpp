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
            std::string bestModel = "";
            int maxGoodMatches;

            // Use Matching class to detect object presence
            bool objectPresence = Matching::findObject(modelDescriptors, modelNames, descTest, matchesThreshold, bestModel, maxGoodMatches);

            // If the object is found, print and save the result
            if (objectPresence) {
                std::cout << "Object '" << key << "' detected in test image: " << ti.name
                          << " (Best model: " << bestModel
                          << " with " << maxGoodMatches << " matches)" << std::endl;
            
              // TODO: Draw bounding box here
            }
        }
    }

    return 0;
}
