#include "facenet_tf.h"
#include "tensorflow/c/c_api.h"


const int scale_h = 160;
const int scale_w = 160;

static void dummy_deallocator(void *data, size_t len, void *arg) {}

string input_layer = "input:0";
string phase_train_layer = "phase_train:0";
string output_layer = "embeddings:0";

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

int main() {
    string model_fname = "/home/zhenglai/models/facenet_20170512-110547.pb";
    string test_img_fname = "/home/zhenglai/repo/tensorflow-mtcnn/cpp/standalone/id0.jpg";
//    string test_img_fname = "/tmp/test.png";
    string feature_vector_fname = "./test.csv";
    TF_Session *sess;
    TF_Graph *graph;
    cv::Mat img_mat = cv::imread(test_img_fname.c_str(), IMREAD_COLOR);

    if (!img_mat.data) {
        std::cerr << "failed to read image file: " << test_img_fname << std::endl;
        return 1;
    }
    cout << img_mat.cols << endl;
    cout << img_mat.channels() << endl;
    cv::cvtColor(img_mat, img_mat, CV_BGR2RGB);


    cv::resize(img_mat, img_mat, cv::Size(scale_w, scale_h), 0, 0);
    img_mat.convertTo(img_mat, CV_32FC3);


    /* tensorflow related*/

    sess = load_graph2(model_fname.c_str(), &graph);
    TF_Status *s = TF_NewStatus();

    std::vector<TF_Input> input_names;
    std::vector<TF_Tensor *> input_values;

    TF_Operation *input_name = TF_GraphOperationByName(graph, input_layer.c_str());
    TF_Operation *phase_train = TF_GraphOperationByName(graph, phase_train_layer.c_str());

    input_names.push_back({input_name, 0});

    const int64_t dim[4] = {1, scale_h, scale_w, 3};

    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT,
                                           dim,
                                           4,
                                           img_mat.ptr(),
                                           sizeof(float) * scale_w * scale_h * 3,
                                           dummy_deallocator,
                                           nullptr);
    TF_Tensor *phase_train_tensor = TF_NewTensor(TF_BOOL,
                                                 new int64_t[]{1},
                                                 1,
                                                 img_mat.ptr(),
                                                 sizeof(float) ,
                                                 dummy_deallocator,
                                                 nullptr);

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


    cout << "tf code" << TF_GetCode(s) << endl;
//    assert(TF_GetCode(s) == TF_OK);

    const float *fvector = (const float *) TF_TensorData(output_values[0]);
    cout << *fvector << endl;

    TF_CloseSession(sess, s);
    TF_DeleteSession(sess, s);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);
}
