// main.cpp
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "detection.hpp"
#include "dataloader.hpp"
#include "matching.hpp"
#include "object_localizer.hpp"

namespace fs = std::filesystem;

int main()
{

    // Define the root path to the dataset
    fs::path rootPath("../data/object_detection_dataset/");

    // Initialize and verify the dataset structure
    FileSystemDataLoader loader;
    if (loader.checkIntegrity(rootPath) != IntegrityCode::OK) {
        std::cerr << "Dataset integrity error\n";
        return -1;
    }
    // Loop over each object category (e.g., sugar_box, mustard_bottle, power_drill)
    for (const auto &key : loader.listObjectKeys(rootPath)) {
        std::cout << "Processing object: " << key << std::endl;

        // Prepare an output directory for annotated results
        fs::path outDir = fs::path("../data/results") / key;
        fs::create_directories(outDir);

        // Load model views for this object
        auto modelViews = loader.loadModelViews(rootPath, key);
        std::vector<cv::Mat>   modelDescriptors;
        std::vector<std::string> modelNames;
        for (auto &mv : modelViews) {
            cv::Mat gray;
            // detect keypoints, compute descriptors
            cv::cvtColor(mv.color, gray, cv::COLOR_BGR2GRAY);
            auto kp   = Detection::detectKeypoints(gray, mv.mask);
            auto desc = Detection::computeDescriptors(gray, kp);
            modelDescriptors.push_back(desc);
            modelNames.push_back(mv.name);
        }

        // process each test image
        for (auto &ti : loader.listTestImages(rootPath, key)) {
            cv::Mat img = cv::imread(ti.path.string());
            if (img.empty()) {
                std::cerr << "  Failed to read " << ti.name << "\n";
                continue;
            }

            // detect keypoints & descriptors on test image
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            auto kpTest   = Detection::detectKeypoints(gray);
            auto descTest = Detection::computeDescriptors(gray, kpTest);

            // determine if object is present using matchesThreshold
            const int matchesThreshold = 8;
            std::string bestModel;
            int maxGoodMatches = 0;
            bool objectPresent = Matching::findObject(
                modelDescriptors,
                modelNames,
                descTest,
                matchesThreshold,
                bestModel,
                maxGoodMatches
            );
            if (!objectPresent) {
                std::cout << "  Object not detected in " << ti.name << "\n";
                continue;
            }

            // fetch top-N view matches
            const int N = 4;
            auto topMatches = Matching::matchTopNModels(modelDescriptors, descTest, N);

            // fuse all matched test points from top-N views
            std::vector<cv::Point2f> fusedPts;
            for (auto &mp : topMatches) {
                int idx = mp.first;
                auto &gm = mp.second;

                cv::Mat grayM;
                cv::cvtColor(modelViews[idx].color, grayM, cv::COLOR_BGR2GRAY);
                auto kpModel = Detection::detectKeypoints(grayM, modelViews[idx].mask);

                auto ptsT = ObjectLocalizer::extractDetectedPoints(kpModel, kpTest, gm);
                fusedPts.insert(fusedPts.end(), ptsT.begin(), ptsT.end());
            }

            // cluster and draw oriented box
            if (!fusedPts.empty()) {
                auto clusterPts = ObjectLocalizer::clusterMeanShift(fusedPts);
                if (!clusterPts.empty()) {
                    ObjectLocalizer::drawBox(img, clusterPts);
                    fs::path savePath = outDir / ("detected_" + ti.name);
                    cv::imwrite(savePath.string(), img);
                    std::cout << "  -> saved with oriented box\n";
                }
            }
        }
    }

    return 0;
}
