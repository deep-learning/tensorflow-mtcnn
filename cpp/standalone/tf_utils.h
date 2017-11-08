#ifndef BIOACK_TF_UTILS_H
#define BIOACK_TF_UTILS_H

#include <tensorflow/c/c_api.h>
#include <cassert>
#include "utils.hpp"

template<size_t SIZE, class T>
inline size_t array_size(T (&arr)[SIZE]) { return SIZE; }

static void dummy_deallocator(void * /*data*/, size_t /*len*/, void * /*arg*/) {}

static TF_Session *tf_load_graph(const char *model_fname, TF_Graph **p_graph, TF_Status *status) {
    TF_Graph *graph = TF_NewGraph();
    std::vector<char> model_buf;
    load_file(model_fname, model_buf);
    TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};
    TF_ImportGraphDefOptions *import_opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
    TF_GraphImportGraphDef(graph, &graph_def, import_opts, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("load graph failed!\n Error: %s\n", TF_Message(status));
        return nullptr;
    }

    TF_SessionOptions *sess_opts = TF_NewSessionOptions();
    TF_Session *session = TF_NewSession(graph, sess_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        printf("load session failed!\n Error: %s\n", TF_Message(status));
        return nullptr;
    }
    *p_graph = graph;
    return session;
}

#endif //BIOACK_TF_UTILS_H
