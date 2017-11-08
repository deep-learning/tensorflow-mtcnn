#include <vector>
#include <algorithm>
#include "jni_types.hpp"
#include "jni_common.hpp"

using namespace std;

FaceDetector::FaceDetector(const char *const model_path) {
    this->handle = 1200;
}

void FaceDetector::release() {
    this->handle = 0L;
}

Face FaceDetector::detect(uchar *img_data, int width, int height, ImageMode img_mode) {
    Point dummy_point = {140, 140};
    std::vector<Point> landmarks(6);
    for (int i = 0; i < landmarks.size(); ++i) {
        landmarks[i] = dummy_point;
    }
//    std::fill_n(landmarks.begin(), landmarks.end(), dummy_point);
    Face dummy_face = {
            .rect = {
                    .x = {100, 100},
                    .width = 200,
                    .height = 300
            },
            .landmarks = landmarks,
            .confidence = 0.82f
    };

    return dummy_face;
}


