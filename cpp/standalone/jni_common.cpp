#include <vector>
#include <algorithm>
#include "jni_types.hpp"
#include "jni_common.hpp"
#include "tf_mtcnn.hpp"

using namespace std;

FaceDetector::FaceDetector(const char *const model_path) {
    auto *ptr = new mtcnn(model_path);
    this->handle = reinterpret_cast<long>(ptr);
}

void FaceDetector::release() {
    auto *ptr = reinterpret_cast<mtcnn *>(this->handle);
    ptr->release();
}

vector<Face> FaceDetector::detect(uchar *img_data, int width, int height, ImageMode img_mode) {
    auto *ptr = reinterpret_cast<mtcnn *>(this->handle);
    auto faces = ptr->detect(img_data, width, height, img_mode);
    vector<Face> result(faces.size());
    for (auto &face: faces) {
        Face f{};
        f.rect = {
                .org = {static_cast<int>(face.x0), static_cast<int>(face.y0)},
                .width = static_cast<int>(face.x1 - face.x0),
                .height = static_cast<int>(face.y1 - face.y0)
        };
        for (int i = 0; i < BA_LANDMARK_TOTAL; ++i) {
            f.landmarks[i] = Point {static_cast<int>(face.landmark.x[i]), static_cast<int>(face.landmark.y[i])};
        }
        f.confidence = face.score;

        result.push_back(f);
    }
    return std::move(result);
}


