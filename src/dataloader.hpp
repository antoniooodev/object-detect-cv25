// DataLoader.hpp
#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>

// Error codes for dataset integrity checks
enum class IntegrityCode
{
    OK = 0,
    InvalidRoot,
    MissingObjectDirs,
    MissingModelsOrTests
};

// Representation of a single model view (color + optional mask)
struct ModelView
{
    cv::Mat color;
    cv::Mat mask;     // empty if no mask available
    std::string name; // base file name without suffix
};

// Representation of a test image (path + file name)
struct TestImage
{
    std::filesystem::path path;
    std::string name;
};

// Abstract interface for dataset loading (supports extension)
class IDataLoader
{
public:
    virtual ~IDataLoader() = default;

    // Verify the directory structure and return an IntegrityCode
    virtual IntegrityCode checkIntegrity(const std::filesystem::path &root) const = 0;

    // List all object keys (e.g. "004_sugar_box") under the root
    virtual std::vector<std::string> listObjectKeys(const std::filesystem::path &root) const = 0;

    // Load all model views (color + mask) for a given object key
    virtual std::vector<ModelView>
    loadModelViews(const std::filesystem::path &root, const std::string &objectKey) const = 0;

    // List all test images (with path and name) for a given object key
    virtual std::vector<TestImage>
    listTestImages(const std::filesystem::path &root, const std::string &objectKey) const = 0;
};

// Concrete filesystem-based loader
class FileSystemDataLoader : public IDataLoader
{
public:
    IntegrityCode checkIntegrity(const std::filesystem::path &root) const override;
    std::vector<std::string> listObjectKeys(const std::filesystem::path &root) const override;
    std::vector<ModelView>
    loadModelViews(const std::filesystem::path &root, const std::string &objectKey) const override;
    std::vector<TestImage>
    listTestImages(const std::filesystem::path &root, const std::string &objectKey) const override;
};

#endif // DATA_LOADER_HPP
