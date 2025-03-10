cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(YoloDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# -------------- OpenCV --------------
set(OpenCV_DIR "F:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- libtorch --------------
list(APPEND CMAKE_PREFIX_PATH "F:/libtorch_251_cu124")
set(Torch_DIR "F:/libtorch_251_cu124/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("${TORCH_LIBRARIES}")
message("${TORCH_INCLUDE_DIRS}")

include_directories(${TORCH_INCLUDE_DIRS})

# -------------- ONNXRuntime ---------------
set(ONNXRUNTIME_ROOTDIR "F:/onnxruntime-win-x64-gpu-1.20.1")
include_directories(YoloDetector ${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB SRCS
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/*.cc
    ${SRC_DIR}/**/*.h
    ${SRC_DIR}/**/*.cpp
    ${SRC_DIR}/**/*.cc
)

add_executable(YoloDetector ${SRCS})
target_link_libraries(YoloDetector ${TORCH_LIBRARIES} ${OpenCV_LIBS} onnxruntime onnxruntime_providers_cuda onnxruntime_providers_tensorrt)
set_property(TARGET YoloDetector PROPERTY CXX_STANDARD 17)
if(MSVC)
    # 获取所有DLL目录路径（分号分隔）
    set(DLL_PATHS 
        "F:/opencv/build/x64/vc16/bin"
        "F:/libtorch_251_cu124/lib"
        "F:/onnxruntime-win-x64-gpu-1.20.1/lib"
    )
    
    # 转换为Windows格式的PATH字符串
    string(REPLACE ";" "\\;" DLL_PATH_STR "${DLL_PATHS}")
    set(ENV_PATH "PATH=%PATH%;${DLL_PATH_STR}")
    
    # 设置VS调试环境变量
    # set_target_properties(YoloDetector PROPERTIES
    #     VS_DEBUGGER_ENVIRONMENT "${ENV_PATH}"
    # )
endif()
