//dataloader.cpp

#include "dataloader.hpp"
#include <iostream>

using Path      = std::filesystem::path;
using DirIter   = std::filesystem::directory_iterator;

IntegrityCode FileSystemDataLoader::checkIntegrity(const Path &root) const {
    if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
        return IntegrityCode::InvalidRoot;
    }
    const std::vector<std::string> objects = {
        "004_sugar_box",
        "006_mustard_bottle",
        "035_power_drill"
    };
    for (const auto &obj : objects) {
        Path objDir = root / obj;
        if (!std::filesystem::is_directory(objDir))
            return IntegrityCode::MissingObjectDirs;
        if (!std::filesystem::is_directory(objDir / "models") || !std::filesystem::is_directory(objDir / "test_images"))
            return IntegrityCode::MissingModelsOrTests;
    }
    return IntegrityCode::OK;
}

std::vector<std::string> FileSystemDataLoader::listObjectKeys(const Path &root) const {
    std::vector<std::string> keys;
    for (auto &entry : DirIter(root)) {
        if (entry.is_directory()) {
            keys.push_back(entry.path().filename().string());
        }
    }
    return keys;
}

std::vector<ModelView>
FileSystemDataLoader::loadModelViews(const Path &root, const std::string &objectKey) const {
    std::vector<ModelView> views;
    Path modelsDir = root / objectKey / "models";

    for (auto &entry : DirIter(modelsDir)) {
        std::string fname = entry.path().filename().string();
        auto pos = fname.find("_color");
        if (pos == std::string::npos) continue;
        std::string base = fname.substr(0, pos);
        std::string ext  = entry.path().extension().string();

        ModelView mv;
        mv.name = base;
        mv.color = cv::imread((modelsDir / (base + "_color" + ext)).string(), cv::IMREAD_COLOR);
        mv.mask  = cv::imread((modelsDir / (base + "_mask" + ext)).string(), cv::IMREAD_GRAYSCALE);
        if (mv.color.empty()) continue;
        views.push_back(mv);
    }
    return views;
}

std::vector<TestImage>
FileSystemDataLoader::listTestImages(const Path &root, const std::string &objectKey) const {
    std::vector<TestImage> files;
    Path testDir = root / objectKey / "test_images";
    for (auto &entry : DirIter(testDir)) {
        if (!entry.is_regular_file()) continue;
        TestImage ti;
        ti.path = entry.path();
        ti.name = entry.path().filename().string();
        files.push_back(ti);
    }
    return files;
}

