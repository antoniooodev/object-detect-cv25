#include <opencv2/opencv.hpp>
#include <filesystem>
#include "preprocessing.hpp"
#include "detection.hpp"
#include "matching.hpp"

namespace fs = std::filesystem;

int main()
{
    // Path to the dataset directory
    std::string datasetPath = "../data/object_detection_dataset/";
    std::string outputFolder = "results/";

    // Create output directory if it doesn't exist
    if (!fs::exists(outputFolder))
    {
        fs::create_directory(outputFolder);
    }

    // Loop through all folders in the dataset
    for (const auto &folder : fs::directory_iterator(datasetPath))
    {
        if (folder.is_directory())
        {
            // Read object folder name
            std::string objectName = folder.path().filename().string();
            std::cout << "Processing object: " << objectName << std::endl;

            // Loop through object test images (update path as needed)
            std::string testImagesPath = folder.path().string() + "/test_images/";
            for (const auto& testImageEntry : fs::directory_iterator(testImagesPath))
            {
                if (testImageEntry.is_regular_file())
                {
                    std::string testImagePath = testImageEntry.path().string();
                    std::string testImageName = testImageEntry.path().filename().string();

                    // Load the test image
                    cv::Mat testImage = Preprocessing::loadImage(testImagePath);
                    if (testImage.empty())
                    {
                        std::cerr << "Error: Could not load image: " << testImagePath << std::endl;
                        continue;
                    }

                    // Preprocess the test image (grayscale + resize)
                    cv::Mat processedTestImage = Preprocessing::preprocessImage(testImage);

                    // Detect test image keypoints using SIFT
                    std::vector<cv::KeyPoint> testKeypoints = Detection::detectKeypoints(processedTestImage);

                    // Compute test image descriptors for the keypoints
                    cv::Mat testDescriptors = Detection::computeDescriptors(processedTestImage, testKeypoints);

                    // Output the results
                    std::cout << "Test Image: " << testImageName << std::endl;
                    std::cout << "Detected keypoints: " << testKeypoints.size() << std::endl;
                    std::cout << "Descriptor matrix size: " << testDescriptors.rows << " x " << testDescriptors.cols << std::endl;

                    // Loop through the  model views for each object ("_color" view names)
                    std::string modelsPath = folder.path().string() + "/models/";
                    for (const auto& modelViewEntry : fs::directory_iterator(modelsPath))
                    {
                        // npos: flag for end of the string; default value returned by find if value not found
                        if (modelViewEntry.is_regular_file() && modelViewEntry.path().filename().string().find("_color") != std::string::npos)
                        {
                            std::string modelViewPath = modelViewEntry.path().string();
                            cv::Mat modelView = cv::imread(modelViewPath, cv::IMREAD_COLOR);
                            if (modelView.empty())
                            {
                                std::cerr << "Error: Could not load model view: " << modelViewPath << std::endl;
                                continue;
                            }

                            // Preprocess the model view (grayscale, noise reduction, Sobel edge detection)
                            cv::Mat processedModelView = Preprocessing::preprocessImage(modelView);

                            // Detect model view keypoints using SIFT
                            std::vector<cv::KeyPoint> modelKeypoints = Detection::detectKeypoints(processedModelView);

                            // Compute model view descriptors for the keypoints
                            cv::Mat modelDescriptors = Detection::computeDescriptors(processedModelView, modelKeypoints);

                            // Match descriptors between the test image and the model view
                            std::vector<cv::DMatch> goodMatches = Matching::matchDescriptors(modelDescriptors, testDescriptors);

                            // Output how many good matches were found
                            std::cout << "Model View: " << modelViewEntry.path().filename().string() << std::endl;
                            std::cout << "Good Matches: " << goodMatches.size() << std::endl;
                        }
                    }
                }
            }
        }
    }

    return 0;

}
