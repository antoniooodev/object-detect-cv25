#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Preprocessing
{
public:
    Preprocessing();
    ~Preprocessing();

    // Load image from file
    static cv::Mat loadImage(const std::string &filename);

    // Preprocess the image (grayscale, noise reduction, edge detection)
    static cv::Mat preprocessImage(const cv::Mat &img);

    // Reduce noise using bilateral filter
    static cv::Mat reduceNoise(const cv::Mat &img);
};

#endif // PREPROCESSING_HPP