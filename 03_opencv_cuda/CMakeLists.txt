cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(opencv_cuda VERSION 1.0)

# Change Debug -> Release for Windows OS
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(CUDA REQUIRED)

# include header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# include OpenCV files
find_package(OpenCV REQUIRED)
include_directories(OpenCV_INCLUDE_DIRS)

# include cuda header files and libraries
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

## add the executable
#file(GLOB SOURCES
#    *.cpp
#)
#add_executable(detect ${SOURCES})

add_executable(detect opencv_cuda.cpp matrix.cpp common.cpp preprocess.cpp)

# link libraries with executable
target_link_libraries(detect ${OpenCV_LIBS})



# visualize all variable used for thorough understanding
message(STATUS "  NVIDIA CUDA:" HAVE_CUDA THEN "YES (ver ${CUDA_VERSION_STRING}, ${cuda_features})" ELSE NO)
message(STATUS "This project directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "openCV libs variables: ${OpenCV_LIBS}")
message(STATUS "openCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "All project sources: ${SOURCES}")

