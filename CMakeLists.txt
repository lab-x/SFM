cmake_minimum_required(VERSION 3.3)
project(SFM)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")

#CUDA
FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_HOST_COMPILER /usr/bin/gcc-4.9)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52)

#OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(usr/local/include; usr/local/cuda)
link_directories(usr/local/lib;usr/local/cuda)



#OpenGL
find_package(GLUT)
find_package(OpenGL)
find_package(GLEW)
if (GLEW_FOUND)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})
    else()
    message("not found")
endif()
include_directories(${FREEGLUT_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIRS})



file(GLOB SOURCE_FILES *.h *.cpp pba/*.cpp pba/*.h pba/*.cu)
#set(SOURCE_FILES main.cpp ViewPair.cpp View.cpp funUtils.cpp Mesh.h SparseCloud.cpp SparseCloud.h)
#add_executable(SFM ${SOURCE_FILES})
cuda_add_executable(SFM ${SOURCE_FILES})

# always after add_executable
target_link_libraries(SFM ${OpenCV_LIBS})
target_link_libraries(SFM ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

