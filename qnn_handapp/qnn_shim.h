// Minimal C ABI for running a single-graph QNN context binary on HTP.
// Inputs/outputs are exposed as float32; quantization to/from UFIXED_POINT_8
// is handled internally using the binary's encoding metadata.

#ifndef QNN_SHIM_H
#define QNN_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qnn_model qnn_model_t;

typedef struct {
    char    name[128];
    int32_t rank;
    int32_t dims[8];
    int32_t qnn_dtype;   // raw Qnn_DataType_t enum value
    float   scale;       // valid for quantized; 1.0 otherwise
    int32_t offset;      // valid for quantized; 0 otherwise
    size_t  num_elements;
} qnn_tensor_info_t;

// Load a context binary and retrieve a single graph.  If graph_name is
// NULL or empty, the first graph in the binary is used.
qnn_model_t* qnn_load(const char* backend_so,
                      const char* system_so,
                      const char* binary_path,
                      const char* graph_name);

int32_t qnn_num_inputs(qnn_model_t* m);
int32_t qnn_num_outputs(qnn_model_t* m);

// Returns pointer to internal tensor info struct.  Lifetime tied to model.
const qnn_tensor_info_t* qnn_input_info(qnn_model_t* m, int32_t idx);
const qnn_tensor_info_t* qnn_output_info(qnn_model_t* m, int32_t idx);

// inputs:  array of float* pointers, one per input tensor (NHWC order, prod(dims) elements)
// outputs: array of float* pointers, one per output tensor (caller-allocated)
// Returns 0 on success, nonzero on failure.
int32_t qnn_execute(qnn_model_t* m, const float* const* inputs, float* const* outputs);

void qnn_destroy(qnn_model_t* m);

// Returns last error string for the model (or static string if none).
const char* qnn_last_error(qnn_model_t* m);

#ifdef __cplusplus
}
#endif

#endif
