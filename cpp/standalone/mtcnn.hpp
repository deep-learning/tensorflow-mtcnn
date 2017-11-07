#ifndef __MTCNN_HPP__
#define __MTCNN_HPP__

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


struct face_landmark {
    float x[5];
    float y[5];
};

struct face_box {
    float x0;
    float y0;
    float x1;
    float y1;

    /* confidence score */
    float score;

    /*regression scale */

    float regress[4];

    /* padding stuff*/
    float px0;
    float py0;
    float px1;
    float py1;

    face_landmark landmark;
};
#endif
