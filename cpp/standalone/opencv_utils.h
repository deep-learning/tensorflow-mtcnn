//
// Created by zhenglai on 17-11-7.
//

#ifndef BIOACK_OPENCV_UTILS_H
#define BIOACK_OPENCV_UTILS_H

#include <opencv2/core.hpp>
#include "dtype.h"

cv::Mat rgb_mat_from(uchar *img_data, int width, int height, ImageMode colorSpaceType);

void resize_mat_in_place(cv::Mat &mat, double scale_factor);

cv::Mat resize_mat(const cv::Mat &mat, double scale_factor);

#endif //BIOACK_OPENCV_UTILS_H
