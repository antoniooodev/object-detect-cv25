// m// main.cpp
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "detection.hpp"
#include "dataloader.hpp"
#include "matching.hpp"
#include "object_localizer.hpp"

namespace fs = std::filesystem;

// Helper to pick box color by object key
static cv::Scalar boxColor(const std::string &key) {
    if (key == "004_sugar_box")        return cv::Scalar(0,   0, 255);  // red
    else if (key == "006_mustard_bottle") return cv::Scalar(255, 0,   0);  // blue
    else if (key == "035_power_drill")    return cv::Scalar(0, 255,   0);  // green
    return cv::Scalar(255, 255, 255); // white fallback
}

int main()
{
    fs::path rootPath("../data/object_detection_dataset/");

    FileSystemDataLoader loader;
    if (loader.checkIntegrity(rootPath) != IntegrityCode::OK) {
        std::cerr << "Dataset integrity error\n";
        return -1;
    }

    // specify per-object match thresholds
    std::map<std::string,int> thresholds = {
        {"004_sugar_box",       20},  // sugar box needs 10 matches
        {"006_mustard_bottle",   12},  // mustard bottle needs 8 matches
        {"035_power_drill",     9}   // power drill needs 12 matches
    };

    // 1) load all object models & descriptors up front
    auto keys = loader.listObjectKeys(rootPath);
    struct ObjData {
        std::string key;
        std::vector<ModelView> views;
        std::vector<cv::Mat> descriptors;
    };
    std::vector<ObjData> allObjects;
    for (auto &key : keys) {
        ObjData od{key};
        od.views = loader.loadModelViews(rootPath, key);
        for (auto &mv : od.views) {
            cv::Mat gray;
            cv::cvtColor(mv.color, gray, cv::COLOR_BGR2GRAY);
            auto kp   = Detection::detectKeypoints(gray, mv.mask);
            auto desc = Detection::computeDescriptors(gray, kp);
            od.descriptors.push_back(desc);
        }
        allObjects.push_back(std::move(od));
    }

    // 2) for each test‐set directory
    for (auto &key : keys) {
        std::cout << "Processing test images in: " << key << std::endl;
        fs::path outDir = fs::path("../data/results/") / key;
        fs::create_directories(outDir);

        auto testImages = loader.listTestImages(rootPath, key);
        for (auto &ti : testImages) {
            cv::Mat img = cv::imread(ti.path.string());
            if (img.empty()) {
                std::cerr << "  Failed to read " << ti.name << "\n";
                continue;
            }

            // detect on test image once
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            auto kpTest   = Detection::detectKeypoints(gray);
            auto descTest = Detection::computeDescriptors(gray, kpTest);

            // try detecting each object
            for (auto &obj : allObjects) {
                // use object-specific threshold
                int matchesThreshold = thresholds[obj.key];

                // quick presence check
                std::string modelName;
                int modelCount;
                bool present = Matching::findObject(
                    obj.descriptors,
                    std::vector<std::string>(obj.views.size(), obj.key),
                    descTest,
                    matchesThreshold,
                    modelName,
                    modelCount
                );
                if (!present)
                    continue;

                // fuse top‐N matches
                const int N = 3;
                auto topMatches = Matching::matchTopNModels(obj.descriptors, descTest, N);

                // collect matched points
                std::vector<cv::Point2f> fusedPts;
                for (auto &mp : topMatches) {
                    int idx = mp.first;
                    auto &gm = mp.second;

                    cv::Mat grayM;
                    cv::cvtColor(obj.views[idx].color, grayM, cv::COLOR_BGR2GRAY);
                    auto kpModel = Detection::detectKeypoints(grayM, obj.views[idx].mask);

                    auto ptsT = ObjectLocalizer::extractDetectedPoints(kpModel, kpTest, gm);
                    fusedPts.insert(fusedPts.end(), ptsT.begin(), ptsT.end());
                }

                // cluster & draw oriented box in object color
                if (!fusedPts.empty()) {
                    auto clusterPts = ObjectLocalizer::clusterMeanShift(fusedPts);
                    if (!clusterPts.empty()) {
                        ObjectLocalizer::drawBox(img, clusterPts, boxColor(obj.key));
                    }
                }
            }

            // save annotated image
            fs::path savePath = outDir / ("detected_" + ti.name);
            cv::imwrite(savePath.string(), img);
            std::cout << "  -> saved " << ti.name << "\n";
        }
    }

    return 0;
}
