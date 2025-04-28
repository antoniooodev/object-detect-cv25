// utils.cpp
#include "utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

namespace fs = std::filesystem;

float computeIoU(const cv::Rect& boxA, const cv::Rect& boxB)
{
    // Compute top left and bottom right coordinates of the boxes
    int xA = std::max(boxA.x, boxB.x);
    int yA = std::max(boxA.y, boxB.y);
    int xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    int yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    // Calculate the two areas
    int boxAArea = boxA.width * boxA.height;
    int boxBArea = boxB.width * boxB.height;

    // Compute IoU
    float iou = static_cast<float>(interArea) / static_cast<float>(boxAArea + boxBArea - interArea);
    return iou;
}

std::vector<GroundTruth> loadAllGroundTruths(const fs::path& rootPath, const std::string& objectKey)
{
    std::vector<GroundTruth> groundTruths;
    fs::path labelsPath = rootPath / objectKey / "labels";

    if (!fs::exists(labelsPath) || !fs::is_directory(labelsPath)) {
        std::cerr << "Labels directory not found: " << labelsPath << "\n";
        return groundTruths;
    }

    for (const auto& entry : fs::directory_iterator(labelsPath)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            if (!file.is_open()) {
                std::cerr << "Failed to open ground truth file: " << entry.path() << "\n";
                continue;
            }
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                std::string id_name;
                int xmin, ymin, xmax, ymax;
                if (!(iss >> id_name >> xmin >> ymin >> xmax >> ymax)) {
                    std::cerr << "Failed to parse ground truth line: " << line << "\n";
                    continue;
                }
                std::string name = id_name.substr(id_name.find('_') + 1); // remove object_id_
                GroundTruth gt;
                gt.objectName = name;
                gt.bbox = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
                groundTruths.push_back(gt);
            }
        }
    }

    return groundTruths;
}

void evaluatePerformance(
    const std::vector<Prediction>& predictions,
    const std::vector<GroundTruth>& groundTruths,
    const std::string& objectKey
)
{
    int correctDetections = 0;
    int totalGroundTruths = 0;
    std::vector<float> ious;

    // Iterate through the predictions, matching each to the corresponding ground truth
    for (const auto& pred : predictions) {
        if (pred.objectName != objectKey)
            continue;

        // Find the corresponding ground truth for this prediction
        auto it = std::find_if(groundTruths.begin(), groundTruths.end(),
                               [&pred](const GroundTruth& gt) {
                                   return gt.objectName == pred.objectName && gt.bbox == pred.bbox;
                               });

        if (it != groundTruths.end()) {
            totalGroundTruths++;

            // Compute IoU between the prediction and the matching ground truth
            float iou = computeIoU(it->bbox, pred.bbox);
            ious.push_back(iou);

            // If IoU is greater than 0.5, count it as a correct detection
            if (iou > 0.5f) {
                correctDetections++;
            }
        }
    }

    // Calculate mIoU (Mean IoU)
    float miou = 0.0f;
    if (!ious.empty()) {
        for (auto iou : ious)
            miou += iou;
        miou /= static_cast<float>(ious.size());
    }

    // Calculate detection accuracy
    float detectionAccuracy = 0.0f;
    if (totalGroundTruths > 0) {
        detectionAccuracy = static_cast<float>(correctDetections) / static_cast<float>(totalGroundTruths);
    }

    std::cout << "\nPerformance for object: " << objectKey << "\n";
    std::cout << "  Mean IoU (mIoU): " << miou << "\n";
    std::cout << "  Detection Accuracy: " << detectionAccuracy * 100.0f << "%\n\n";
}


