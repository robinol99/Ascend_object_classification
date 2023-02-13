cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
#set(TensorRT_DIR /home/robinol/Downloads/TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.3.1)

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(TensorRT_LIB /usr/lib/x86_64-linux-gnu)
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)

link_directories(${TensorRT_LIB})
link_libraries(nvinfer)

# We require CUDA, OpenCV
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
message(${TensorRT_INCLUDE_DIRS})


find_library(_NVINFER_LIB nvinfer nvonnxparser HINTS ${TensorRT_LIB} PATH_SUFFIXES lib lib64 REQUIRED)
set(NVINFER_LIB ${_NVINFER_LIB})

add_executable(object_classification_preprocess src/object_classification_preprocess.cpp src/utilities/utilities_nvidia.cpp)
target_include_directories(object_classification_preprocess PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${TensorRT_DIR}/samples/common)
target_link_libraries(object_classification_preprocess PUBLIC nvonnxparser ${NVINFER_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(inference_test src/inference_test.cpp src/utilities/utilities_nvidia.cpp)
target_include_directories(inference_test PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${TensorRT_DIR}/samples/common)
target_link_libraries(inference_test PUBLIC nvonnxparser ${NVINFER_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
