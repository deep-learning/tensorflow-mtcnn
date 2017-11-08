#ifndef BIOACK_JNI_COMMON_HPP
#define BIOACK_JNI_COMMON_HPP

#include <vector>
#include "jni_types.hpp"

class FaceDetector {
public:
    FaceDetector(const char *const model_path);

    ~FaceDetector() { this->release(); }

    void release();

    Face detect(uchar *img_data, int width, int height, ImageMode imageMode);

private:
    long handle;
};


class FaceRecognizer {
public:
private:
};

#endif //BIOACK_JNI_COMMON_HPP
