#include "facenet_tf.h"
#include "tensorflow/c/c_api.h"
#include "tf_utils.h"


const int scale_h = 160;
const int scale_w = 160;


string input_layer = "input";
string phase_train_layer = "phase_train";
string output_layer = "embeddings";



void dump_mat(Mat mat) {
    std::cout << "(" << mat.cols << ", " << mat.rows << ", " << mat.channels() << ")" << endl;
}

static void BoolDeallocator(void *data, size_t, void *arg) {
    delete[] static_cast<bool *>(data);
}

static TF_Tensor *boolTensor(bool v) {
    const int num_bytes = sizeof(bool);
    cout << "num_bytes=" << num_bytes << endl;
    bool *values = new bool[1];
    values[0] = v;
    return TF_NewTensor(TF_BOOL, nullptr, 0, values, num_bytes, &BoolDeallocator, nullptr);
}


const float *get_feature_vector(Mat img_mat) {
    string model_fname = "/home/zhenglai/models/facenet_20170512-110547.pb";
    if (!img_mat.data) {
        std::cerr << "img_mat invalid" << endl;
        return nullptr;
    }
    cout << img_mat.cols << endl;
    cout << img_mat.channels() << endl;
    cv::cvtColor(img_mat, img_mat, CV_BGR2RGB);


    cv::resize(img_mat, img_mat, cv::Size(scale_w, scale_h), 0, 0);
    dump_mat(img_mat);
    img_mat.convertTo(img_mat, CV_32FC3);

    Mat mat(scale_w, scale_h, CV_32FC3);

    Scalar mean, stddev;
    meanStdDev(img_mat, mean, stddev);
    cout << "mean=" << mean[0] << ", stddev=" << stddev[0] << endl;
    mat.setTo(mean);
    mat = img_mat - mat;
    mat.convertTo(mat, CV_32FC3, 1 / stddev[0]);

    cout << "val=" << mat.at<float>(12, 12) << endl;
    dump_mat(mat);


    TF_Session *sess = nullptr;
    TF_Graph *graph = nullptr;

    TF_Status *s = TF_NewStatus();
    sess = tf_load_graph(model_fname.c_str(), &graph, s);

    assert(sess != nullptr);
    assert(graph != nullptr);

    std::vector<TF_Output> input_names;
    std::vector<TF_Tensor *> input_values;

    TF_Operation *input_name = TF_GraphOperationByName(graph, input_layer.c_str());
    TF_Operation *phase_train = TF_GraphOperationByName(graph, phase_train_layer.c_str());

    assert(input_name != nullptr);
    assert(phase_train != nullptr);

    input_names.push_back({input_name, 0});
    input_names.push_back({phase_train, 0});

    const int64_t dim[4] = {1, scale_h, scale_w, 3};

    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT,
                                           dim,
                                           4,
                                           mat.ptr(),
                                           sizeof(float) * scale_w * scale_h * 3,
                                           dummy_deallocator,
                                           nullptr);


    TF_Tensor *phase_train_tensor = boolTensor(false);

    input_values.push_back(input_tensor);
    input_values.push_back(phase_train_tensor);

    std::vector<TF_Output> output_names;

    TF_Operation *output_name = TF_GraphOperationByName(graph, output_layer.c_str());
    output_names.push_back({output_name, 0});

    std::vector<TF_Tensor *> output_values(output_names.size(), nullptr);


    cout << "starting to run" << endl;
    TF_SessionRun(sess,
                  nullptr,
                  input_names.data(), input_values.data(), input_names.size(),
                  output_names.data(), output_values.data(), output_names.size(),
                  nullptr,
                  0,
                  nullptr,
                  s);

    if (TF_GetCode(s) != TF_OK) {
        cerr << TF_Message(s) << endl;
    }

    const float *fvector = (const float *) TF_TensorData(output_values[0]);

    TF_CloseSession(sess, s);
    TF_DeleteSession(sess, s);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);
    return fvector;
}

int main(int argc, char **argv) {
    const float *fa = get_feature_vector(imread(argv[1]));
    const float *fb = get_feature_vector(imread(argv[2]));
//    cv::Mat img_mat = cv::imread(test_img_fname.c_str(), IMREAD_COLOR);
    cout << "calculating" << endl;
    cout << cosine_similarity(fa, fb, 128) << endl;

//    Mat img_mat = Mat(160, 160, CV_8UC3);
//    randu(img_mat, Scalar::all(0), Scalar::all(255));


}
