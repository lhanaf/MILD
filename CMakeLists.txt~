cmake_minimum_required(VERSION 3.2)
project(MILD)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

file(GLOB helloworld_SRC
"*.cpp"
"*.h"
"MILD/*.cpp"
"MILD/*.h"
"ORBSLAM/*.cpp"
"ORBSLAM/*.h"
)

set(SOURCE_FILES
 ${helloworld_SRC})

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(OpenCV 3 REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "-msse4.2") 

add_executable(MILD ${SOURCE_FILES})


include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(MILD ${OpenCV_LIBS} )
