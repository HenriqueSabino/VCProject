cmake_minimum_required(VERSION 3.0.0)
project(VCProject VERSION 0.1.0)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/Users/hsabi/Documents/Rural/5P/VC/opencv/build/x64/vc15/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(VCProject main.cpp)

target_link_libraries(VCProject ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
