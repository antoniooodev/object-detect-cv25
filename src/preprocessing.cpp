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

// Preprocess the image (convert to grayscale, reduce noise, extract edges)
cv::Mat Preprocessing::preprocessImage(const cv::Mat &img)
{
    cv::Mat gray, denoised, edges;

    // Convert to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Reduce noise
    denoised = reduceNoise(gray);

    // Apply Sobel filter for edge detection
    edges = applySobel(denoised);

    return edges; // Return image with edges detected
}

// Apply Sobel filter for edge detection
cv::Mat Preprocessing::applySobel(const cv::Mat &img)
{
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;

    // Sobel filter for horizontal and vertical gradients
    cv::Sobel(img, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_16S, 0, 1, 3);

    // Convert gradients to absolute values
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Combine the gradients
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return grad;
}

// Reduce noise using bilateral filter
cv::Mat Preprocessing::reduceNoise(const cv::Mat &img)
{
    cv::Mat denoised;

    // Apply bilateral filter to reduce noise while preserving edges
    cv::bilateralFilter(img, denoised, 9, 75, 75);

    return denoised;
}