#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Box> Utils::loadGroundTruthBoxes(const std::string& filename) {
    std::vector<Box> boxes;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open ground truth file: " << filename << std::endl;
        return boxes;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Box box;
        int xmin, ymin, xmax, ymax;
        if (!(iss >> box.label >> xmin >> ymin >> xmax >> ymax)) {
            continue; // skip invalid lines
        }
        box.bbox = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
        boxes.push_back(box);
    }
    return boxes;
}

float Utils::computeIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    int interArea = (boxA & boxB).area();
    int unionArea = boxA.area() + boxB.area() - interArea;
    if (unionArea <= 0) return 0.0f;
    return static_cast<float>(interArea) / unionArea;
}
