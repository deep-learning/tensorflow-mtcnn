CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(bioack)

set(CMAKE_CXX_STANDARD 14)

set(TF_ROOT ./tf140)
set(OPENCV_ROOT ./opencv330)

INCLUDE_DIRECTORIES(${TF_ROOT}/include ${OPENCV_ROOT}/include)
LINK_DIRECTORIES(${TF_ROOT}/lib ${OPENCV_ROOT}/lib)

SET(SOURCE_FILES
        mtcnn.hpp
#        camera.cpp
        test.cpp
        tensorflow_mtcnn.cpp
        #        facenet_tf.cpp
        comm_lib.cpp
        utils.cpp
        #        camera.cpp
        facenet.cpp
        facenet.h
        dtype.h
        opencv_utils.h
        opencv_utils.cpp
        tf_utils.h tf_mtcnn.cpp tf_mtcnn.h jni_types.hpp jni_common.cpp jni_common.hpp)

add_executable(bioack ${SOURCE_FILES})
target_link_libraries(bioack tensorflow opencv_core opencv_imgcodecs opencv_imgproc)
