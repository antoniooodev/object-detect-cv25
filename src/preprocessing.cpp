#include "preprocessing.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

Preprocessing::Preprocessing() {}

Preprocessing::~Preprocessing() {}

// Load an image from file
cv::Mat Preprocessing::loadImage(const std::string &filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Error: Unable to load image!" << std::endl;
    }
    return img;
}

// Preprocess the image (convert to grayscale, reduce noise)
cv::Mat Preprocessing::preprocessImage(const cv::Mat &img)
{
    cv::Mat gray, denoised, edges;

    // Convert to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Reduce noise
    denoised = reduceNoise(gray);

    return denoised; // Return image with edges detected
}

// Reduce noise using bilateral filter
cv::Mat Preprocessing::reduceNoise(const cv::Mat &img)
{
    cv::Mat denoised;

    // Apply bilateral filter to reduce noise while preserving edges
    cv::bilateralFilter(img, denoised, 9, 75, 75);

    return denoised;
}