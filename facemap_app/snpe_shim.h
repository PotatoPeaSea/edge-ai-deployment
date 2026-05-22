// Minimal C ABI for running a single DLC on the SNPE runtime (HTP or CPU)
// in-process via libSNPE.so.  Inputs/outputs are exchanged as float32; SNPE
// performs quantization to/from the DLC's fixed-point encoding internally.

#ifndef SNPE_SHIM_H
#define SNPE_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct snpe_model snpe_model_t;

typedef struct {
    char    name[128];
    int32_t rank;
    int32_t dims[8];
    size_t  num_elements;
} snpe_tensor_info_t;

// Load a DLC and build an SNPE instance.
//   use_dsp          : 1 -> run on Hexagon HTP (DSP), 0 -> CPU float32.
//   accelerated_init : 1 -> enable HTP accelerated init (offline-prepared DLC).
//   output_names     : comma-separated output tensor names to expose, or NULL/""
//                      to use SNPE's default (final output only).  Required when
//                      a network has multiple outputs (e.g. detector scores+boxes).
// Returns a model handle.  Even on failure a non-null handle may be returned;
// always check snpe_last_error() == "ok".
snpe_model_t* snpe_load(const char* dlc_path, int use_dsp, int accelerated_init,
                        const char* output_names);

int32_t snpe_num_inputs(snpe_model_t* m);
int32_t snpe_num_outputs(snpe_model_t* m);

// Pointer to internal tensor info struct; lifetime tied to the model.
const snpe_tensor_info_t* snpe_input_info(snpe_model_t* m, int32_t idx);
const snpe_tensor_info_t* snpe_output_info(snpe_model_t* m, int32_t idx);

// inputs:  array of float* (one per input tensor, prod(dims) elements each)
// outputs: array of float* (caller-allocated, one per output tensor)
// Returns 0 on success, nonzero on failure.
int32_t snpe_execute(snpe_model_t* m, const float* const* inputs, float* const* outputs);

void snpe_destroy(snpe_model_t* m);

const char* snpe_last_error(snpe_model_t* m);

#ifdef __cplusplus
}
#endif

#endif
