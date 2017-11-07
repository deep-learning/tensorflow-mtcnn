CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(bioack)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(OpenCV REQUIRED)

SET(SOURCE_FILES
        tensorflow/c/c_api.h
#        tensorflow_mtcnn.cpp
        facenet_tf.cpp
        comm_lib.cpp
        utils.cpp
#        camera.cpp
        facenet.cpp facenet.h dtype.h opencv_utils.h opencv_utils.cpp)

add_executable(bioack ${SOURCE_FILES})
target_link_libraries(bioack tensorflow ${OpenCV_LIBS})
