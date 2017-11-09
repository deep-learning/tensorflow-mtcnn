
#include <fstream>
#include <utility>
#include <vector>


#include "tensorflow/c/c_api.h"
#include "tensorflow_mtcnn.hpp"
#include "mtcnn.hpp"
#include "utils.hpp"
#include "comm_lib.hpp"
#include "tf_utils.hpp"
#include "tf_mtcnn.hpp"

#include <unistd.h>

using std::string;

int main(int argc, char *argv[]) {
    string image = "./imgs/test.jpg";
    string model_fname = "./models/mtcnn_frozen_model.pb";
    string output_fname = "./tmp/test_new.jpg";
    int save_chop = 0;
    int res;

    while ((res = getopt(argc, argv, "i:o:m:s")) != -1) {
        switch (res) {
            case 'i':
                image = optarg;
                break;
            case 'o':
                output_fname = optarg;
                break;
            case 's':
                save_chop = 1;
                break;
            case 'm':
                model_fname = optarg;
                break;
            default:
                break;
        }
    }

//    TF_Session *sess;
//    TF_Graph *graph;
//    TF_Status *status = TF_NewStatus();
//
//    sess = tf_load_graph(model_fname.c_str(), &graph, status);
//
//    if (sess == nullptr)
//        return 1;

    //Load image

    cv::Mat frame = cv::imread(image);

    if (!frame.data) {
        std::cerr << "failed to read image file: " << image << std::endl;
        return 1;
    }


    mtcnn *detector = new mtcnn(model_fname.c_str());

//    unsigned long start_time = now();

//    mtcnn_detect(sess, graph, frame, face_info);

//    unsigned long end_time = now();

    std::vector<face_box> face_info = detector->detect(frame);

    for (int i = 0; i < face_info.size(); i++) {
        face_box &box = face_info[i];

        printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n", i,
               box.x0, box.y0, box.x1, box.y1, box.score);
        printf("landmark: ");

        for (unsigned int j = 0; j < 5; j++)
            printf(" (%2.5f %2.5f)", box.landmark.x[j], box.landmark.y[j]);

        printf("\n");


        if (save_chop) {

            cv::Mat corp_img = frame(cv::Range(box.y0, box.y1),
                                     cv::Range(box.x0, box.x1));

            char title[128];
            sprintf(title, "/tmp/id%d.jpg", i);
            cv::imwrite(title, corp_img);
        }

        /*draw box */

        cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);


        /* draw landmark */

        for (int l = 0; l < 5; l++) {
            cv::circle(frame, cv::Point(box.landmark.x[l], box.landmark.y[l]), 1, cv::Scalar(0, 0, 255), 2);

        }
    }

    cv::imwrite(output_fname, frame);

    std::cout << "boxed faces are in file: " << output_fname << std::endl;

//    TF_Status *s = TF_NewStatus();
//
//    TF_CloseSession(sess, s);
//    TF_DeleteSession(sess, s);
//    TF_DeleteGraph(graph);
//    TF_DeleteStatus(s);

    return 0;
}







