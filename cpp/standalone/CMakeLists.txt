CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(bioack)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})

message(STATUS ${OpenCV_INCLUDE_DIR})
message(STATUS ${OpenCV_LIBS})

SET(SOURCE_FILES
#        tensorflow_mtcnn.cpp
        facenet_tf.cpp
        comm_lib.cpp
        utils.cpp
#        camera.cpp
        )

add_executable(bioack ${SOURCE_FILES})
target_link_libraries(bioack tensorflow ${OpenCV_LIBS})
