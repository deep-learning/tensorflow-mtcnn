
#include <opencv2/opencv.hpp>
#include "jni_common.hpp"
#include "jni_types.hpp"

using namespace std;
using namespace cv;

int main() {
    const char *img_fname = "./imgs/test.jpg";
    const char *model_fname = "./models/mtcnn_frozen_model.pb";
    const char *output_fname = "./tmp/test_new.jpg";
    cv::Mat frame = cv::imread(img_fname);

    if (!frame.data) {
        std::cerr << "failed to read image file: " << img_fname << std::endl;
        return 1;
    }

    auto *ptr = new FaceDetector(model_fname);
    auto faces = ptr->detect(frame.data, frame.cols, frame.rows, RGB);

    for (int i = 0; i < faces.size(); i++) {
        auto &box = faces[i];

        printf("face %d: x0,y0 %2.5f %2.5f  w,h %2.5f  %2.5f conf: %2.5f\n", i,
               box.rect.org.x, box.rect.org.y, box.rect.width, box.rect.height, box.confidence);
        printf("landmark: ");

        for (unsigned int j = 0; j < 5; j++)
            printf(" (%2.5f %2.5f)", box.landmarks[j].x, box.landmarks[j].y);

        printf("\n");


//        cv::Mat corp_img = frame(cv::Range(box.y0, box.y1),
//                                 cv::Range(box.x0, box.x1));
//
//        char title[128];
//        sprintf(title, "/tmp/id%d.jpg", i);
//        cv::imwrite(title, corp_img);

        /*draw box */

        cv::rectangle(frame,
                      cv::Point(box.rect.org.x, box.rect.org.y),
                      cv::Point(box.rect.org.x + box.rect.width, box.rect.height),
                      cv::Scalar(0, 255, 0),
                      1);


        /* draw landmark */

        for (int l = 0; l < 5; l++) {
            cv::circle(frame, cv::Point(box.landmarks[l].x, box.landmarks[l].y), 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::imwrite(output_fname, frame);

    std::cout << "boxed faces are in file: " << output_fname << std::endl;

}

