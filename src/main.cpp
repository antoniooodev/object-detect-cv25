//author: Giuseppe D'Auria, Paolo Conti, Antonio Tangaro

#include <opencv2/opencv.hpp>
#include <filesystem>
#include "detection.hpp"
#include "dataloader.hpp"
#include "matching.hpp"
#include "object_localizer.hpp"
#include "utils.hpp"

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

    // Specify per-object match thresholds
    std::map<std::string,int> thresholds = {
        {"004_sugar_box",       20},
        {"006_mustard_bottle",   12},
        {"035_power_drill",      9}
    };

    // Counters for evaluation
    std::map<std::string, int> truePositives, totalGroundTruths;
    std::map<std::string, float> iouSums;
    std::map<std::string, int> iouCounts;

    // Load all object models & descriptors
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

    // Process each test set
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

            // Detect on test image
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            auto kpTest   = Detection::detectKeypoints(gray);
            auto descTest = Detection::computeDescriptors(gray, kpTest);

            // Store predicted boxes
            std::vector<Box> predictedBoxes;

            // Try detecting each object
            for (auto &obj : allObjects) {
                int matchesThreshold = thresholds[obj.key];

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

                // Fuse top‐N matches
                const int N = 3;
                auto topMatches = Matching::matchTopNModels(obj.descriptors, descTest, N);

                // Collect matched points
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

                // Cluster and draw oriented box
                if (!fusedPts.empty()) {
                    auto clusterPts = ObjectLocalizer::clusterMeanShift(fusedPts);
                    if (!clusterPts.empty()) {
                        ObjectLocalizer localizer;
                        auto rbox = localizer.drawBox(img, clusterPts, boxColor(obj.key));
                        if (rbox.size.width > 0 && rbox.size.height > 0) {
                            cv::Rect bbox = rbox.boundingRect() & cv::Rect(0, 0, img.cols, img.rows);
                            predictedBoxes.push_back({obj.key, bbox});
                        }
                    }
                }
            }

            // Save annotated image
            fs::path savePath = outDir / ("detected_" + ti.name);
            cv::imwrite(savePath.string(), img);
            std::cout << "  -> saved " << ti.name << "\n";

            // Load ground truth boxes
            std::string gtFile = ti.path.string();

            // Go up two levels and add 'labels' folder
            std::filesystem::path gtPath = std::filesystem::path(gtFile).parent_path().parent_path() / "labels";

            // Extract the filename without the extension and remove the "-color" part
            // Get filename
            std::string filename = gtFile.substr(gtFile.find_last_of('/') + 1);
            // Find the last dot (before extension)
            size_t dotPos = filename.find_last_of('.');
            if (dotPos != std::string::npos) {
                // Remove the extension
                filename = filename.substr(0, dotPos);
            }

            // Remove the "-color" part if it exists
            size_t colorPos = filename.find("-color");
            if (colorPos != std::string::npos) {
                // Erase the "-color" part (6 characters
                filename.erase(colorPos, 6);
            }
            
            // Add "-box.txt" to the base filename
            gtPath /= filename + "-box.txt";

            auto gtBoxes = Utils::loadGroundTruthBoxes(gtPath);



            // Count total ground truths
            for (const auto &gt : gtBoxes) {
                totalGroundTruths[gt.label]++;
            }

            // Match predictions to ground truth
            for (const auto &gt : gtBoxes) {
                float bestIoU = 0.0f;
                for (const auto &pred : predictedBoxes) {
                    if (pred.label == gt.label) {
                        float iou = Utils::computeIoU(pred.bbox, gt.bbox);
                        if (iou > bestIoU)
                            bestIoU = iou;
                    }
                }
                if (bestIoU > 0.5f)
                    truePositives[gt.label]++;

                iouSums[gt.label] += bestIoU;
                iouCounts[gt.label]++;
            }
        }
    }

    // Print evaluation results (mIoU and true positives found)
    std::cout << "\n=== Detection Results ===\n";
    for (const auto &[label, total] : totalGroundTruths) {
        int tp = truePositives[label];
        float miou = (iouCounts[label] > 0) ? (iouSums[label] / iouCounts[label]) : 0.0f;
        std::cout << "Object: " << label << "\n";
        std::cout << "  True Positives: " << tp << " / " << total << "\n";
        std::cout << "  Mean IoU: " << miou << "\n";
    }

    return 0;
}
