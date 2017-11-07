#ifndef BIOACK_TF_UTILS_H
#define BIOACK_TF_UTILS_H

#include <tensorflow/c/c_api.h>
#include "utils.hpp"


void dummy_deallocator(void * /*data*/, size_t /*len*/, void * /*arg*/) {}

TF_Session *tf_load_graph(const char *model_fname, TF_Graph **p_graph) {
    TF_Status *s = TF_NewStatus();
    TF_Graph *graph = TF_NewGraph();
    std::vector<char> model_buf;
    load_file2(model_fname, model_buf);
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

#endif //BIOACK_TF_UTILS_H
