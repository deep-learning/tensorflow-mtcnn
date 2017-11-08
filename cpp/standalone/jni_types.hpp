#ifndef BIOACK_JNI_TYPES_HPP
#define BIOACK_JNI_TYPES_HPP

#include <cstddef>
#include <vector>

typedef unsigned char uchar;

enum ImageMode {
    RGB,
    YUV
};

typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    Point x;
    int width;
    int height;
} Rect;

typedef struct {
    Rect rect;
    std::vector<Point> landmarks;
    float confidence;
} Face;

// landmark index
const int BA_EYE_LEFT_IX = 0;
const int BA_EYE_RIGHT_IX = 1;
const int BA_NOSE_TIP_IX = 2;
const int BA_MOUSE_LEFT_CORNER_IX = 3;
const int BA_MOUSE_RIGHT_CORNER_IX = 4;

#endif //BIOACK_COMMON_HPP