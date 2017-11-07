#ifndef __TENSORFLOW_MTCNN_HPP__
#define __TENSORFLOW_MTCNN_HPP_

#include "tensorflow/c/c_api.h"
#include <opencv2/opencv.hpp>
#include "mtcnn.hpp"

void mtcnn_detect(TF_Session* sess, TF_Graph * graph, const cv::Mat& img, std::vector<face_box>& face_list);

#endif
