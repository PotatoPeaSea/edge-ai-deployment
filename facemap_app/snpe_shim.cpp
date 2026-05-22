// In-process SNPE runtime shim.  Loads a DLC and runs it on HTP (or CPU)
// through libSNPE.so's C API, keeping the SNPE instance alive across calls so
// per-frame cost is the actual inference time, not subprocess startup.
//
// Build (on the aarch64 device, where libSNPE.so lives):
//   g++ -O2 -fPIC -shared -std=c++17 -I<sdk>/include/SNPE \
//       snpe_shim.cpp -o libsnpe_shim.so \
//       -L<sdk>/lib -lSNPE
//
// I/O is float32; SNPE quantizes to the DLC's fixed-point encoding internally.

#include "snpe_shim.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "SNPE/SNPE.h"
#include "SNPE/SNPEBuilder.h"
#include "SNPE/SNPEUtil.h"
#include "DlContainer/DlContainer.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/TensorMap.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/StringList.h"
#include "DlSystem/RuntimeList.h"

struct snpe_model {
    Snpe_DlContainer_Handle_t container = nullptr;
    Snpe_SNPE_Handle_t        snpe      = nullptr;

    // Persistent input tensors + map, reused every execute.
    std::vector<Snpe_ITensor_Handle_t> input_tensors;
    Snpe_TensorMap_Handle_t input_map  = nullptr;
    Snpe_TensorMap_Handle_t output_map = nullptr;

    std::vector<std::string>       input_names;
    std::vector<std::string>       output_names;
    std::vector<snpe_tensor_info_t> input_infos;
    std::vector<snpe_tensor_info_t> output_infos;

    std::string last_error = "ok";
};

#define SET_ERR(m, msg) do { (m)->last_error = (msg); \
    fprintf(stderr, "[snpe_shim] %s\n", (msg)); } while (0)

namespace {

snpe_tensor_info_t make_info(const char* name, Snpe_TensorShape_Handle_t shape) {
    snpe_tensor_info_t info{};
    if (name) strncpy(info.name, name, sizeof(info.name) - 1);
    size_t rank = Snpe_TensorShape_Rank(shape);
    if (rank > 8) rank = 8;
    info.rank = (int32_t)rank;
    size_t elems = 1;
    for (size_t i = 0; i < rank; ++i) {
        size_t d = Snpe_TensorShape_At(shape, i);
        info.dims[i] = (int32_t)d;
        elems *= d;
    }
    info.num_elements = elems;
    return info;
}

} // namespace

extern "C" {

const char* snpe_last_error(snpe_model_t* m) {
    return m ? m->last_error.c_str() : "model is null";
}

int32_t snpe_num_inputs(snpe_model_t* m)  { return m ? (int32_t)m->input_infos.size()  : 0; }
int32_t snpe_num_outputs(snpe_model_t* m) { return m ? (int32_t)m->output_infos.size() : 0; }

const snpe_tensor_info_t* snpe_input_info(snpe_model_t* m, int32_t i) {
    if (!m || i < 0 || i >= (int32_t)m->input_infos.size()) return nullptr;
    return &m->input_infos[i];
}
const snpe_tensor_info_t* snpe_output_info(snpe_model_t* m, int32_t i) {
    if (!m || i < 0 || i >= (int32_t)m->output_infos.size()) return nullptr;
    return &m->output_infos[i];
}

void snpe_destroy(snpe_model_t* m) {
    if (!m) return;
    if (m->input_map)  Snpe_TensorMap_Delete(m->input_map);
    if (m->output_map) Snpe_TensorMap_Delete(m->output_map);
    for (auto t : m->input_tensors) if (t) Snpe_ITensor_Delete(t);
    if (m->snpe)      Snpe_SNPE_Delete(m->snpe);
    if (m->container) Snpe_DlContainer_Delete(m->container);
    delete m;
}

snpe_model_t* snpe_load(const char* dlc_path, int use_dsp, int accelerated_init,
                        const char* output_names) {
    auto* m = new snpe_model_t();

    m->container = Snpe_DlContainer_Open(dlc_path);
    if (!m->container) { SET_ERR(m, "Snpe_DlContainer_Open failed"); return m; }

    Snpe_SNPEBuilder_Handle_t builder = Snpe_SNPEBuilder_Create(m->container);
    if (!builder) { SET_ERR(m, "Snpe_SNPEBuilder_Create failed"); return m; }

    Snpe_RuntimeList_Handle_t runtimes = Snpe_RuntimeList_Create();
    if (use_dsp) {
        Snpe_RuntimeList_Add(runtimes, SNPE_RUNTIME_DSP_FIXED8_TF);
    }
    // CPU is always added as a fallback so a build never hard-fails.
    Snpe_RuntimeList_Add(runtimes, SNPE_RUNTIME_CPU_FLOAT32);
    Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder, runtimes);
    Snpe_SNPEBuilder_SetPerformanceProfile(builder, SNPE_PERFORMANCE_PROFILE_BURST);
    if (accelerated_init) {
        Snpe_SNPEBuilder_SetAcceleratedInit(builder, true);
    }

    // Multiple-output networks (e.g. detector scores+boxes) need their output
    // tensors named explicitly, else SNPE exposes only the final one.
    Snpe_StringList_Handle_t out_list = nullptr;
    if (output_names && output_names[0]) {
        out_list = Snpe_StringList_Create();
        std::string s(output_names), tok;
        size_t start = 0;
        while (start <= s.size()) {
            size_t comma = s.find(',', start);
            if (comma == std::string::npos) comma = s.size();
            tok = s.substr(start, comma - start);
            if (!tok.empty()) Snpe_StringList_Append(out_list, tok.c_str());
            start = comma + 1;
        }
        Snpe_SNPEBuilder_SetOutputTensors(builder, out_list);
    }

    m->snpe = Snpe_SNPEBuilder_Build(builder);
    if (out_list) Snpe_StringList_Delete(out_list);
    Snpe_RuntimeList_Delete(runtimes);
    Snpe_SNPEBuilder_Delete(builder);
    if (!m->snpe) {
        std::string e = "Snpe_SNPEBuilder_Build failed: ";
        e += Snpe_Util_GetLastError();
        SET_ERR(m, e.c_str());
        return m;
    }

    // ---- inputs: names, shapes, persistent ITensors -------------------------
    Snpe_StringList_Handle_t in_names = Snpe_SNPE_GetInputTensorNames(m->snpe);
    if (!in_names) { SET_ERR(m, "GetInputTensorNames failed"); return m; }
    size_t n_in = Snpe_StringList_Size(in_names);
    m->input_map = Snpe_TensorMap_Create();
    for (size_t i = 0; i < n_in; ++i) {
        const char* nm = Snpe_StringList_At(in_names, i);
        m->input_names.emplace_back(nm ? nm : "");
        Snpe_TensorShape_Handle_t shape = Snpe_SNPE_GetInputDimensions(m->snpe, nm);
        m->input_infos.push_back(make_info(nm, shape));
        Snpe_ITensor_Handle_t t = Snpe_Util_CreateITensor(shape);  // zero-init float32
        Snpe_TensorShape_Delete(shape);
        m->input_tensors.push_back(t);
        Snpe_TensorMap_Add(m->input_map, m->input_names.back().c_str(), t);
    }
    Snpe_StringList_Delete(in_names);

    // ---- outputs: names known up front; sizes discovered via a dummy run ----
    Snpe_StringList_Handle_t out_names = Snpe_SNPE_GetOutputTensorNames(m->snpe);
    if (!out_names) { SET_ERR(m, "GetOutputTensorNames failed"); return m; }
    size_t n_out = Snpe_StringList_Size(out_names);
    for (size_t i = 0; i < n_out; ++i) {
        const char* nm = Snpe_StringList_At(out_names, i);
        m->output_names.emplace_back(nm ? nm : "");
    }
    Snpe_StringList_Delete(out_names);

    m->output_map = Snpe_TensorMap_Create();
    if (Snpe_SNPE_ExecuteITensors(m->snpe, m->input_map, m->output_map) != SNPE_SUCCESS) {
        std::string e = "warmup ExecuteITensors failed: ";
        e += Snpe_Util_GetLastError();
        SET_ERR(m, e.c_str());
        return m;
    }
    for (size_t i = 0; i < m->output_names.size(); ++i) {
        Snpe_ITensor_Handle_t t =
            Snpe_TensorMap_GetTensor_Ref(m->output_map, m->output_names[i].c_str());
        if (!t) { SET_ERR(m, "output tensor missing after warmup"); return m; }
        Snpe_TensorShape_Handle_t shape = Snpe_ITensor_GetShape(t);
        m->output_infos.push_back(make_info(m->output_names[i].c_str(), shape));
        Snpe_TensorShape_Delete(shape);
    }

    m->last_error = "ok";
    return m;
}

int32_t snpe_execute(snpe_model_t* m, const float* const* inputs, float* const* outputs) {
    if (!m || !m->snpe) { if (m) SET_ERR(m, "execute on null/invalid model"); return -1; }

    // Copy caller floats into the persistent input tensors.
    for (size_t i = 0; i < m->input_tensors.size(); ++i) {
        void* dst = Snpe_ITensor_GetData(m->input_tensors[i]);
        if (!dst) { SET_ERR(m, "ITensor_GetData returned null"); return -2; }
        memcpy(dst, inputs[i], m->input_infos[i].num_elements * sizeof(float));
    }

    Snpe_TensorMap_Clear(m->output_map);
    if (Snpe_SNPE_ExecuteITensors(m->snpe, m->input_map, m->output_map) != SNPE_SUCCESS) {
        std::string e = "ExecuteITensors failed: ";
        e += Snpe_Util_GetLastError();
        SET_ERR(m, e.c_str());
        return -3;
    }

    for (size_t i = 0; i < m->output_names.size(); ++i) {
        Snpe_ITensor_Handle_t t =
            Snpe_TensorMap_GetTensor_Ref(m->output_map, m->output_names[i].c_str());
        if (!t) { SET_ERR(m, "output tensor missing"); return -4; }
        const void* src = Snpe_ITensor_GetData(t);
        memcpy(outputs[i], src, m->output_infos[i].num_elements * sizeof(float));
    }

    return 0;
}

} // extern "C"
