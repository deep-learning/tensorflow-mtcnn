//
// Created by zhenglai on 17-11-7.
//

#ifndef BIOACK_TF_MTCNN_H
#define BIOACK_TF_MTCNN_H

#include <string>
#include <opencv2/core.hpp>

#include "tensorflow/c/c_api.h"
#include "mtcnn.hpp"
#include "dtype.h"

class mtcnn2 {
public:
    mtcnn2(void) {
        min_size_ = 40;
        pnet_threshold_ = 0.6;
        rnet_threshold_ = 0.7;
        onet_threshold_ = 0.9;
        factor_ = 0.709;

    }

    void set_threshold(float p, float r, float o) {
        pnet_threshold_ = p;
        rnet_threshold_ = r;
        onet_threshold_ = o;
    }


    virtual int load_model(const std::string &model_dir)=0;

    virtual void detect(cv::Mat &img, std::vector<face_box> &face_list)=0;

    virtual ~mtcnn2(void) {};

protected:

    int min_size_;
    float pnet_threshold_;
    float rnet_threshold_;
    float onet_threshold_;
    float factor_;
};

/* factory part */

class mtcnn_factory {
public:

    typedef mtcnn2 *(*creator)(void);

    static void register_creator(const std::string &name, creator &create_func);

    static mtcnn2 *create_detector(const std::string &name);

    static std::vector<std::string> list(void);

private:
    mtcnn_factory() {};


};

class only_for_auto_register {
public:
    only_for_auto_register(std::string name, mtcnn_factory::creator func) {
        mtcnn_factory::register_creator(name, func);
    }

};

#define REGISTER_MTCNN_CREATOR(name, func) \
    static  only_for_auto_register dummy_mtcnn_creator_## name (#name, func)


class mtcnn {
public:

    mtcnn(const char *model_fname);

    ~mtcnn();

    void set_threshold(float p, float r, float o) {
        this->pnet_threshold = p;
        this->rnet_threshold = r;
        this->onet_threshold = o;
    }

    std::vector<cv::Point> detect(const cv::Mat &mat);

    std::vector<cv::Point> detect(uchar *img_data, int width, int height, ImageMode img_mode);

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
