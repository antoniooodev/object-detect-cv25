# Object Detection CV25

## Project Objective

The goal of this project is to develop an object detection system in C++ using OpenCV. The system will identify and classify objects in complex images, specifically focusing on three objects: mustard bottles, power drills, and sugar boxes.

## Requirements

- OpenCV 4.x
- C++ compiler (e.g., GCC, Clang)
- CMake (for build configuration)

## How to Run

1. Clone the repository:

```bash
   git clone https://github.com/yourusername/object-detect-cv25.git
   cd object-detect-cv25
```

2. Install the dependencies (OpenCV):

- For Ubuntu: `sudo apt-get install libopencv-dev`
- For Windows: Follow instructions at https://opencv.org/releases/

3. Build the project using CMake:

```bash
   mkdir build

   cd build

   cmake ..

   make
```

5. Run the program:

```bash
   ./object_detect
```

## Project Structure

- `src/`: Contains the main C++ source code for the project
- `data/`: Directory containing the object detection dataset
- `include/`: External library headers (e.g., OpenCV)
- `README.md`: This file

## Dataset

The dataset is provided as a zip file and contains images for three objects: mustard bottles, power drills, and sugar boxes. These images are located under the `data/object_detection_dataset` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
