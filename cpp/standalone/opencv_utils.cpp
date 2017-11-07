//
// Created by zhenglai on 17-11-7.
//

#include <opencv2/imgproc.hpp>
#include "opencv_utils.h"

using namespace cv;

// todo: optimize this..
cv::Mat rgb_mat_from(uchar *img_data, int width, int height, ColorSpaceType colorSpaceType) {
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

