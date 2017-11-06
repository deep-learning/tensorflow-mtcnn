#include "facenet_tf.h"
#include "tensorflow/c/c_api.h"


const int scale_h = 160;
const int scale_w = 160;

static void dummy_deallocator(void *data, size_t len, void *arg) {}

string input_layer = "input";
string phase_train_layer = "phase_train";
string output_layer = "embeddings";


static int load_file2(const std::string &fname, std::vector<char> &buf) {
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


TF_Session *load_graph2(const char *frozen_fname, TF_Graph **p_graph) {
    TF_Status *s = TF_NewStatus();

    TF_Graph *graph = TF_NewGraph();

    std::vector<char> model_buf;

    load_file2(frozen_fname, model_buf);

    TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

    TF_ImportGraphDefOptions *import_opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
    TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

    if (TF_GetCode(s) != TF_OK) {
        printf("load graph failed!\n Error: %s\n", TF_Message(s));

        return nullptr;
    }

    TF_SessionOptions *sess_opts = TF_NewSessionOptions();
    TF_Session *session = TF_NewSession(graph, sess_opts, s);
    assert(TF_GetCode(s) == TF_OK);


    TF_DeleteStatus(s);


    *p_graph = graph;

    return session;
}

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

float cosine_similarity(const float * const A, const float * const B, unsigned int Vector_Length) {
    float dot = 0.0f, denom_a = 0.0f, denom_b = 0.0f;
    for (unsigned int i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}


int main() {
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    string model_fname = "/home/zhenglai/models/facenet_20170512-110547.pb";
    string test_img_fname = "/home/zhenglai/repo/tensorflow-mtcnn/cpp/standalone/id0.jpg";
//    cv::Mat img_mat = cv::imread(test_img_fname.c_str(), IMREAD_COLOR);

    Mat img_mat = Mat(160, 160, CV_8UC3);
    randu(img_mat, Scalar::all(0), Scalar::all(255));

    if (!img_mat.data) {
        std::cerr << "failed to read image file: " << test_img_fname << std::endl;
        return 1;
    }
    cout << img_mat.cols << endl;
    cout << img_mat.channels() << endl;
    cv::cvtColor(img_mat, img_mat, CV_BGR2RGB);


    cv::resize(img_mat, img_mat, cv::Size(scale_w, scale_h), 0, 0);
    dump_mat(img_mat);
    img_mat.convertTo(img_mat, CV_32FC3);
    cout << img_mat.at<float>(10, 10) << endl;
    dump_mat(img_mat);

    /* tensorflow related*/

    TF_Session *sess = nullptr;
    TF_Graph *graph = nullptr;
    sess = load_graph2(model_fname.c_str(), &graph);
    TF_Status *s = TF_NewStatus();

    assert(sess != nullptr);
    assert(graph != nullptr);

    std::vector<TF_Output> input_names;
    std::vector<TF_Tensor *> input_values;

    TF_Operation *input_name = TF_GraphOperationByName(graph, input_layer.c_str());
    TF_Operation *phase_train = TF_GraphOperationByName(graph, phase_train_layer.c_str());

    assert(input_name != nullptr);
    assert(phase_train != nullptr);

    input_names.push_back({phase_train, 0});
    input_names.push_back({input_name, 0});

    const int64_t dim[4] = {1, scale_h, scale_w, 3};

    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT,
                                           dim,
                                           4,
                                           img_mat.ptr(),
                                           sizeof(float) * scale_w * scale_h * 3,
                                           dummy_deallocator,
                                           nullptr);


    TF_Tensor *phase_train_tensor = boolTensor(false);

    input_values.push_back(phase_train_tensor);
    input_values.push_back(input_tensor);

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

//    for (size_t i = 0; i < 128; ++i) {
//        cout << fvector[i] << ',';
//    }

    cout << cosine_similarity(fvector, fvector, 128) << endl;


    TF_CloseSession(sess, s);
    TF_DeleteSession(sess, s);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);
}
