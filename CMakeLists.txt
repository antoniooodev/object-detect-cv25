cmake_minimum_required(VERSION 3.10)
project(object-detect-cv25)

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(NOT COMPILER_SUPPORTS_CXX17)
    message(FATAL_ERROR "C++17 or filesystem support is required. Boost filesystem will be used instead.")
    find_package(Boost REQUIRED COMPONENTS filesystem)
    add_definitions(-DUSE_BOOST_FILESYSTEM)
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(object-detect 
    src/main.cpp 
    src/detection.cpp
    src/preprocessing.cpp
    src/matching.cpp
    src/object_localizer.cpp
    src/dataloader.cpp
)

target_link_libraries(object-detect ${OpenCV_LIBS})

enable_testing()
add_subdirectory(tests)
