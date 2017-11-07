#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <sys/time.h>

int load_file(const std::string &fname, std::vector<char> &buf) {
    std::ifstream fs(fname, std::ios::binary | std::ios::in);
    if (!fs.good()) {
        std::cerr << fname << " does not exist" << std::endl;
        return -1;
    }

    fs.seekg(0, std::ios::end);
    int fsize = fs.tellg();
    fs.seekg(0, std::ios::beg);
    buf.resize(fsize);
    fs.read(buf.data(), fsize);
    fs.close();
    return 0;
}

unsigned long now(void) {
    struct timeval tv;
    unsigned long ts;
    gettimeofday(&tv, NULL);
    ts = tv.tv_sec * 1000000 + tv.tv_usec;
    return ts;
}

float cosine_similarity(const float *const A, const float *const B, size_t Vector_Length) {
    float dot = 0.0F, denom_a = 0.0F, denom_b = 0.0F;
    for (size_t i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

void save_float(const char *name, const float *data, int size) {
    char fname[128];

    sprintf(fname, "%s", name);

    std::cout << "save data to " << fname << "   size " << size << std::endl;
    std::ofstream of;
    of.open(fname);

    for (int i = 0; i < size; i++) {
        of << std::setprecision(6) << data[i] << "," << std::endl;
    }

    of.close();
}


void save_img(const char *name, void *p_img) {
    const cv::Mat &img = *(cv::Mat *) p_img;
    int row = img.rows;
    int col = img.cols;
    int chan = img.channels();

    int sz = row * col * chan;
    char fname[128];

    int data;

    sprintf(fname, "%s", name);

    std::cout << "save data to " << fname << "   size " << sz << std::endl;
    std::ofstream of;
    of.open(fname);


    col = col * chan;

    if (img.isContinuous()) {
        col = col * row;
        row = 1;
    }

    for (int i = 0; i < row; i++) {
        const unsigned char *p = img.ptr<unsigned char>(i);

        for (int j = 0; j < col; j++) {
            data = p[j];

            of << data << "," << std::endl;
        }
    }

    of.close();
}



