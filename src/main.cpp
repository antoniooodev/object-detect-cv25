#include <opencv2/opencv.hpp>
#include <filesystem>
#include "preprocessing.hpp"
#include "detection.hpp"

namespace fs = std::__fs::filesystem;

int main()
{
    // Path to the dataset directory
    std::string datasetPath = "../data/object_detection_dataset/004_sugar_box/test_images/";
    std::string outputFolder = "results/";

    // Create output directory if it doesn't exist
    if (!fs::exists(outputFolder))
    {
        fs::create_directory(outputFolder);
    }

    // Loop through all image files in the dataset folder
    for (const auto &entry : fs::directory_iterator(datasetPath))
    {
        if (entry.is_regular_file())
        {
            std::string imagePath = entry.path().string();
            std::string imageName = entry.path().filename().string();

            // Load the image
            cv::Mat img = Preprocessing::loadImage(imagePath);
            if (img.empty())
            {
                std::cerr << "Error: Could not load image: " << imagePath << std::endl;
                continue;
            }

            // Preprocess the image (grayscale + resize)
            cv::Mat processedImage = Preprocessing::preprocessImage(img);

            // Detect keypoints using SIFT
            std::vector<cv::KeyPoint> keypoints = Detection::detectKeypoints(processedImage);

            // Compute descriptors for the keypoints
            cv::Mat descriptors = Detection::computeDescriptors(processedImage, keypoints);

            // Output the results
            std::cout << "Image: " << imageName << std::endl;
            std::cout << "Detected keypoints: " << keypoints.size() << std::endl;
            std::cout << "Descriptor matrix size: " << descriptors.rows << " x " << descriptors.cols << std::endl;

            // Draw keypoints on the image
            cv::Mat outputImage;
            cv::drawKeypoints(processedImage, keypoints, outputImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            // Save the result with keypoints drawn
            std::string outputImagePath = outputFolder + "sift_" + imageName;
            cv::imwrite(outputImagePath, outputImage);
        }
    }

    return 0;
}
