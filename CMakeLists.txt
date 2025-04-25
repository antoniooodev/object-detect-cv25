cmake_minimum_required(VERSION 3.10)
project(object-detect-cv25)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(object-detect 
    src/main.cpp 
    src/preprocessing.cpp
    src/detection.cpp
    src/matching.cpp
    src/dataloader.cpp
)

target_link_libraries(object-detect ${OpenCV_LIBS})

