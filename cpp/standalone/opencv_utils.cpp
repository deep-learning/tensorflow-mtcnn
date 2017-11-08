#include "opencv2/imgproc.hpp"
#include "opencv_utils.h"

using namespace cv;

// todo: optimize this..
cv::Mat rgb_mat_from(uchar *const img_data, int width, int height, ImageMode colorSpaceType) {
    switch (colorSpaceType) {
        case RGB: {
            Mat mat(height, width, CV_8UC3, img_data);
            return std::move(mat);
        }
        case YUV: {
            Mat yuv_mat(height + height / 2, width, CV_8UC1, img_data);
            Mat rgbaMat;
            cvtColor(yuv_mat, rgbaMat, COLOR_YUV2RGBA_NV21, 4);
            Mat rgbMat;
            cvtColor(rgbaMat, rgbMat, COLOR_RGBA2RGB, 3);
            return std::move(rgbMat);
        }
    }
}

void resize_mat_in_place(cv::Mat &mat, double scale_factor) {
    if (scale_factor > 1.0) { // todo float arith ops.
        cv::resize(mat, mat, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
    } else if (scale_factor < 1.0) {
#ifdef FAST
        cv::resize(mat, mat, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
#else
        cv::resize(mat, mat, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);
#endif
    } else {
        // ignore, do nothing
    }
}

cv::Mat resize_mat(const cv::Mat &mat, double scale_factor) {
    Mat ret = mat.clone();
    resize_mat_in_place(ret, scale_factor);
    return ret;
}

