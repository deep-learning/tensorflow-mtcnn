//
// Created by zhenglai on 17-11-7.
//

#ifndef BIOACK_TF_MTCNN_H
#define BIOACK_TF_MTCNN_H

#include <string>
#include "opencv2/core.hpp"

#include "tensorflow/c/c_api.h"
#include "mtcnn.hpp"
#include "jni_types.hpp"

class mtcnn {
public:

    mtcnn(const char *model_fname);

    ~mtcnn() { this->release(); }

    void release();

    void set_threshold(float p, float r, float o) {
        this->pnet_threshold = p;
        this->rnet_threshold = r;
        this->onet_threshold = o;
    }

    std::vector<face_box> detect(const cv::Mat &mat);

    std::vector<face_box> detect(uchar *img_data, int width, int height, ImageMode img_mode);

    void set_factor(float factor) {
        this->factor = factor;
    }

    void set_min_size(float min_size) {
        this->min_size = min_size;
    }

private:
    TF_Graph *graph;
    TF_Session *session;
    TF_Status *status;
    int min_size;
    float pnet_threshold;
    float rnet_threshold;
    float onet_threshold;
    float factor;
};

#endif //BIOACK_TF_MTCNN_H
