//
// Created by zhenglai on 17-11-7.
//

#include "tf_mtcnn.hpp"
#include "tf_utils.hpp"
#include "tensorflow_mtcnn.hpp"
#include "tensorflow/c/c_api.h"
#include "opencv_utils.hpp"

using namespace std;
using namespace cv;

mtcnn::mtcnn(const char *model_fname) {
    this->min_size = 40;
    this->pnet_threshold = 0.6;
    this->rnet_threshold = 0.7;
    this->onet_threshold = 0.9;
    this->factor = 0.709;
    this->status = TF_NewStatus();
    this->session = tf_load_graph(model_fname, &(this->graph), this->status);
}

std::vector<face_box> mtcnn::detect(const cv::Mat &mat) {
    if (!mat.data) {
        std::cerr << "failed to read mat data: " << mat << std::endl;
        return std::vector<face_box>();
    }
    std::vector<face_box> face_info;
    unsigned long start = now();


    mtcnn_detect(this->session, this->graph, mat, face_info);

    unsigned long end = now();
#ifdef DEBUG
    std::cout << "total detected: " << face_info.size() << " faces. used " << (end - start) << " us"
              << std::endl;
    for (auto &box : face_info) {
        printf("face: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n",
               box.x0, box.y0, box.x1, box.y1, box.score);
        printf("landmark: ");
        for (size_t j = 0; j < 5; j++)
            printf(" (%2.5f %2.5f)", box.landmark.x[j], box.landmark.y[j]);
        printf("\n");
    }
#endif
    return face_info;
}

std::vector<face_box> mtcnn::detect(uchar *const img_data, int width, int height, ImageMode img_mode) {
    Mat mat = rgb_mat_from(img_data, width, height, img_mode);
    return move(this->detect(mat));
}

void mtcnn::release() {
    if (this->session != nullptr) {
        TF_CloseSession(this->session, this->status);
        TF_DeleteSession(this->session, this->status);
    }

    if (this->status != nullptr) {
        TF_DeleteGraph(this->graph);
    }

    if (this->status != nullptr) {
        TF_DeleteStatus(this->status);
    }
}
