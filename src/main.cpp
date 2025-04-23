#include <opencv2/opencv.hpp>
#include "preprocessing.hpp"
#include "detection.hpp"

int main()
{
    // Load the image
    cv::Mat img = Preprocessing::loadImage("../data/object_detection_dataset/004_sugar_box/test_images/4_0049_000003-color.jpg");

    if (img.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Preprocess the image (grayscale + resize)
    cv::Mat processedImage = Preprocessing::preprocessImage(img);

    // Save the preprocessed image
    cv::imwrite("processed_image.jpg", processedImage);

    // Detect keypoints using SIFT
    std::vector<cv::KeyPoint> keypoints = Detection::detectKeypoints(processedImage);

    // Compute descriptors for the keypoints
    cv::Mat descriptors = Detection::computeDescriptors(processedImage, keypoints);

    // Output the results
    std::cout << "Detected keypoints: " << keypoints.size() << std::endl;
    std::cout << "Descriptor matrix size: " << descriptors.rows << " x " << descriptors.cols << std::endl;

    // Draw keypoints on the image
    cv::Mat outputImage;
    cv::drawKeypoints(processedImage, keypoints, outputImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Show the result
    cv::imshow("SIFT Keypoints", outputImage);
    cv::waitKey(0);

    // Save the result with keypoints drawn
    cv::imwrite("sift_keypoints.jpg", outputImage);

    return 0;
}
