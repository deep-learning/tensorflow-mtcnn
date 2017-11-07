#include <fstream>
#include <utility>
#include <vector>


#include "tensorflow/c/c_api.h"
#include "tensorflow_mtcnn.hpp"
#include "utils.hpp"
#include "tf_utils.h"

using std::string;

int main2(int argc, char *argv[]) {
    string model_fname = "./models/mtcnn_frozen_model.pb";

    cv::VideoCapture camera;

    camera.open(0);

    if (!camera.isOpened()) {
        std::cerr << "failed to open camera" << std::endl;
        return 1;
    }


    TF_Session *sess;
    TF_Graph *graph;

    TF_Status *s = TF_NewStatus();

    sess = tf_load_graph(model_fname.c_str(), &graph, s);

    if (sess == nullptr)
        return 1;


    cv::Mat frame;


    while (1) {

        camera.read(frame);

        std::vector<face_box> face_info;

        unsigned long start_time = now();

        mtcnn_detect(sess, graph, frame, face_info);

        unsigned long end_time = now();


        for (unsigned int i = 0; i < face_info.size(); i++) {
            face_box &box = face_info[i];

            /*draw box */

            cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);


            /* draw landmark */

            for (int l = 0; l < 5; l++) {
                cv::circle(frame, cv::Point(box.landmark.x[l], box.landmark.y[l]), 1, cv::Scalar(0, 0, 255), 2);

            }
        }

        std::cout << "total detected: " << face_info.size() << " faces. used " << (end_time - start_time) << " us"
                  << std::endl;

        cv::imshow("camera", frame);

        cv::waitKey(5);
    }


    TF_CloseSession(sess, s);
    TF_DeleteSession(sess, s);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);

    return 0;
}







