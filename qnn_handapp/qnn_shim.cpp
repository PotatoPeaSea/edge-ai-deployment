// Minimal QNN runtime shim for AoT context binaries on HTP.
//
// Compile against the QAIRT/QNN headers shipped with the SDK; links only
// against libdl (QNN libs are dlopened at runtime).
//
// Build:
//   g++ -O2 -fPIC -shared -std=c++17 -I<sdk>/include/QNN \
//       qnn_shim.cpp -o libqnn_shim.so -ldl

#include "qnn_shim.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

namespace {

// ------------- Tensor field helpers (V1/V2) -------------------------------

#define TGET(t, field, V1, V2) \
    ((t).version == QNN_TENSOR_VERSION_2 ? (t).v2.field : (t).v1.field)

static inline Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
}
static inline uint32_t tensor_rank(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}
static inline uint32_t* tensor_dims(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}
static inline const char* tensor_name(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name;
}
static inline Qnn_QuantizeParams_t tensor_qparams(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.quantizeParams : t.v1.quantizeParams;
}

static inline void tensor_set_membuf(Qnn_Tensor_t& t, void* data, uint32_t size_bytes) {
    if (t.version == QNN_TENSOR_VERSION_2) {
        t.v2.memType = QNN_TENSORMEMTYPE_RAW;
        t.v2.clientBuf.data = data;
        t.v2.clientBuf.dataSize = size_bytes;
    } else {
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data = data;
        t.v1.clientBuf.dataSize = size_bytes;
    }
}

static size_t dtype_size_bytes(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32: return 4;
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: return 2;
        case QNN_DATATYPE_FLOAT_64:
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_INT_64:           return 8;
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8:           return 1;
        default: return 0;
    }
}

static bool is_quantized(Qnn_DataType_t dt) {
    return dt == QNN_DATATYPE_UFIXED_POINT_8 ||
           dt == QNN_DATATYPE_SFIXED_POINT_8 ||
           dt == QNN_DATATYPE_UFIXED_POINT_16 ||
           dt == QNN_DATATYPE_SFIXED_POINT_16 ||
           dt == QNN_DATATYPE_UFIXED_POINT_32 ||
           dt == QNN_DATATYPE_SFIXED_POINT_32;
}

// Pull scale/offset from a QuantizeParams (scale-offset only; trivial path).
static void extract_quant(const Qnn_QuantizeParams_t& qp, float& scale, int32_t& offset) {
    scale = 1.0f;
    offset = 0;
    if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        scale  = qp.scaleOffsetEncoding.scale;
        offset = qp.scaleOffsetEncoding.offset;
    }
    // axis-quant / block-quant not handled — would need per-element scale/offset
}

// ------------- Deep copy of tensor descriptor (no client buffer) ----------

static bool deep_copy_tensor(Qnn_Tensor_t& dst, const Qnn_Tensor_t& src) {
    dst.version = src.version;
    if (src.version == QNN_TENSOR_VERSION_1) {
        dst.v1 = src.v1;
        if (src.v1.rank > 0 && src.v1.dimensions) {
            uint32_t* d = (uint32_t*)malloc(sizeof(uint32_t) * src.v1.rank);
            memcpy(d, src.v1.dimensions, sizeof(uint32_t) * src.v1.rank);
            dst.v1.dimensions = d;
        }
        if (src.v1.name) dst.v1.name = strdup(src.v1.name);
        dst.v1.memType = QNN_TENSORMEMTYPE_RAW;
        dst.v1.clientBuf.data = nullptr;
        dst.v1.clientBuf.dataSize = 0;
    } else {
        dst.v2 = src.v2;
        if (src.v2.rank > 0 && src.v2.dimensions) {
            uint32_t* d = (uint32_t*)malloc(sizeof(uint32_t) * src.v2.rank);
            memcpy(d, src.v2.dimensions, sizeof(uint32_t) * src.v2.rank);
            dst.v2.dimensions = d;
        }
        if (src.v2.name) dst.v2.name = strdup((char*)src.v2.name);
        dst.v2.memType = QNN_TENSORMEMTYPE_RAW;
        dst.v2.clientBuf.data = nullptr;
        dst.v2.clientBuf.dataSize = 0;
        dst.v2.isDynamicDimensions = nullptr;
    }
    return true;
}

static void free_tensor_owned(Qnn_Tensor_t& t) {
    if (t.version == QNN_TENSOR_VERSION_1) {
        free(t.v1.dimensions); t.v1.dimensions = nullptr;
        free((void*)t.v1.name); t.v1.name = nullptr;
        free(t.v1.clientBuf.data); t.v1.clientBuf.data = nullptr;
    } else {
        free(t.v2.dimensions); t.v2.dimensions = nullptr;
        free((void*)t.v2.name); t.v2.name = nullptr;
        free(t.v2.clientBuf.data); t.v2.clientBuf.data = nullptr;
    }
}

// ------------- Provider/interface lookups ---------------------------------

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn)(const QnnSystemInterface_t***, uint32_t*);

} // namespace

// =============================================================================
// Model state
// =============================================================================

struct qnn_model {
    void* backend_lib = nullptr;
    void* system_lib  = nullptr;

    QNN_INTERFACE_VER_TYPE iface{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sys_iface{};

    Qnn_BackendHandle_t  backend = nullptr;
    Qnn_DeviceHandle_t   device  = nullptr;
    Qnn_ContextHandle_t  context = nullptr;
    Qnn_GraphHandle_t    graph   = nullptr;

    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    std::vector<qnn_tensor_info_t> input_infos;
    std::vector<qnn_tensor_info_t> output_infos;

    std::string last_error = "ok";
};

#define SET_ERR(m, msg) do { (m)->last_error = (msg); fprintf(stderr, "[qnn_shim] %s\n", (msg)); } while (0)

extern "C" {

const char* qnn_last_error(qnn_model_t* m) {
    return m ? m->last_error.c_str() : "model is null";
}

int32_t qnn_num_inputs(qnn_model_t* m)  { return m ? (int32_t)m->inputs.size()  : 0; }
int32_t qnn_num_outputs(qnn_model_t* m) { return m ? (int32_t)m->outputs.size() : 0; }

const qnn_tensor_info_t* qnn_input_info(qnn_model_t* m, int32_t i) {
    if (!m || i < 0 || i >= (int32_t)m->input_infos.size()) return nullptr;
    return &m->input_infos[i];
}
const qnn_tensor_info_t* qnn_output_info(qnn_model_t* m, int32_t i) {
    if (!m || i < 0 || i >= (int32_t)m->output_infos.size()) return nullptr;
    return &m->output_infos[i];
}

void qnn_destroy(qnn_model_t* m) {
    if (!m) return;
    if (m->context && m->iface.contextFree) {
        m->iface.contextFree(m->context, nullptr);
    }
    if (m->device && m->iface.deviceFree) {
        m->iface.deviceFree(m->device);
    }
    if (m->backend && m->iface.backendFree) {
        m->iface.backendFree(m->backend);
    }
    for (auto& t : m->inputs)  free_tensor_owned(t);
    for (auto& t : m->outputs) free_tensor_owned(t);
    if (m->backend_lib) dlclose(m->backend_lib);
    if (m->system_lib)  dlclose(m->system_lib);
    delete m;
}

static qnn_tensor_info_t make_info(const Qnn_Tensor_t& t) {
    qnn_tensor_info_t info{};
    const char* nm = tensor_name(t);
    if (nm) {
        strncpy(info.name, nm, sizeof(info.name) - 1);
    }
    info.rank = (int32_t)tensor_rank(t);
    uint32_t* dims = tensor_dims(t);
    size_t elems = 1;
    for (int i = 0; i < info.rank && i < 8; ++i) {
        info.dims[i] = (int32_t)dims[i];
        elems *= dims[i];
    }
    info.num_elements = elems;
    info.qnn_dtype = (int32_t)tensor_dtype(t);
    Qnn_QuantizeParams_t qp = tensor_qparams(t);
    extract_quant(qp, info.scale, info.offset);
    return info;
}

qnn_model_t* qnn_load(const char* backend_so,
                      const char* system_so,
                      const char* binary_path,
                      const char* graph_name)
{
    auto* m = new qnn_model_t();

    // 1. dlopen backend + system libs
    m->backend_lib = dlopen(backend_so, RTLD_NOW | RTLD_GLOBAL);
    if (!m->backend_lib) { SET_ERR(m, dlerror()); return m; }
    m->system_lib = dlopen(system_so, RTLD_NOW | RTLD_LOCAL);
    if (!m->system_lib)  { SET_ERR(m, dlerror()); return m; }

    auto get_iface = (QnnInterfaceGetProvidersFn)dlsym(m->backend_lib, "QnnInterface_getProviders");
    auto get_sys   = (QnnSystemInterfaceGetProvidersFn)dlsym(m->system_lib, "QnnSystemInterface_getProviders");
    if (!get_iface || !get_sys) { SET_ERR(m, "missing getProviders symbol"); return m; }

    {
        const QnnInterface_t** providers = nullptr;
        uint32_t n = 0;
        if (get_iface(&providers, &n) != QNN_SUCCESS || n == 0 || !providers) {
            SET_ERR(m, "QnnInterface_getProviders failed"); return m;
        }
        bool found = false;
        for (uint32_t i = 0; i < n; ++i) {
            if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
                providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
                m->iface = providers[i]->QNN_INTERFACE_VER_NAME;
                found = true; break;
            }
        }
        if (!found) { SET_ERR(m, "no compatible backend interface"); return m; }
    }
    {
        const QnnSystemInterface_t** providers = nullptr;
        uint32_t n = 0;
        if (get_sys(&providers, &n) != QNN_SUCCESS || n == 0 || !providers) {
            SET_ERR(m, "QnnSystemInterface_getProviders failed"); return m;
        }
        bool found = false;
        for (uint32_t i = 0; i < n; ++i) {
            if (providers[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
                providers[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
                m->sys_iface = providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
                found = true; break;
            }
        }
        if (!found) { SET_ERR(m, "no compatible system interface"); return m; }
    }

    // 2. read binary file
    std::vector<uint8_t> buf;
    {
        struct stat st{};
        if (stat(binary_path, &st) != 0) { SET_ERR(m, "context binary not found"); return m; }
        buf.resize(st.st_size);
        FILE* fp = fopen(binary_path, "rb");
        if (!fp) { SET_ERR(m, "fopen binary failed"); return m; }
        if (fread(buf.data(), 1, st.st_size, fp) != (size_t)st.st_size) {
            fclose(fp); SET_ERR(m, "fread binary failed"); return m;
        }
        fclose(fp);
    }

    // 3. system context → binary info → tensor metadata
    QnnSystemContext_Handle_t sys_ctx = nullptr;
    if (m->sys_iface.systemContextCreate(&sys_ctx) != QNN_SUCCESS) {
        SET_ERR(m, "systemContextCreate failed"); return m;
    }
    const QnnSystemContext_BinaryInfo_t* bin_info = nullptr;
    Qnn_ContextBinarySize_t bin_info_size = 0;
    if (m->sys_iface.systemContextGetBinaryInfo(sys_ctx, buf.data(), buf.size(),
                                                 &bin_info, &bin_info_size) != QNN_SUCCESS) {
        SET_ERR(m, "systemContextGetBinaryInfo failed");
        m->sys_iface.systemContextFree(sys_ctx);
        return m;
    }

    // Pick the requested graph (or first one).
    uint32_t num_graphs = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
    if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        num_graphs = bin_info->contextBinaryInfoV1.numGraphs;
        graphs     = bin_info->contextBinaryInfoV1.graphs;
    } else if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        num_graphs = bin_info->contextBinaryInfoV2.numGraphs;
        graphs     = bin_info->contextBinaryInfoV2.graphs;
    } else if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        num_graphs = bin_info->contextBinaryInfoV3.numGraphs;
        graphs     = bin_info->contextBinaryInfoV3.graphs;
    }
    if (num_graphs == 0 || !graphs) {
        SET_ERR(m, "no graphs in binary");
        m->sys_iface.systemContextFree(sys_ctx);
        return m;
    }

    int chosen = 0;
    if (graph_name && graph_name[0]) {
        chosen = -1;
        for (uint32_t g = 0; g < num_graphs; ++g) {
            const char* gn = nullptr;
            if (graphs[g].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) gn = graphs[g].graphInfoV1.graphName;
            else if (graphs[g].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) gn = graphs[g].graphInfoV2.graphName;
            else if (graphs[g].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) gn = graphs[g].graphInfoV3.graphName;
            if (gn && strcmp(gn, graph_name) == 0) { chosen = (int)g; break; }
        }
        if (chosen < 0) {
            SET_ERR(m, "graph_name not found in binary");
            m->sys_iface.systemContextFree(sys_ctx);
            return m;
        }
    }

    uint32_t n_in = 0, n_out = 0;
    Qnn_Tensor_t* gin = nullptr;
    Qnn_Tensor_t* gout = nullptr;
    const char* chosen_name = nullptr;
    const QnnSystemContext_GraphInfo_t& gi = graphs[chosen];
    if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        n_in = gi.graphInfoV1.numGraphInputs;  gin = gi.graphInfoV1.graphInputs;
        n_out = gi.graphInfoV1.numGraphOutputs; gout = gi.graphInfoV1.graphOutputs;
        chosen_name = gi.graphInfoV1.graphName;
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        n_in = gi.graphInfoV2.numGraphInputs;  gin = gi.graphInfoV2.graphInputs;
        n_out = gi.graphInfoV2.numGraphOutputs; gout = gi.graphInfoV2.graphOutputs;
        chosen_name = gi.graphInfoV2.graphName;
    } else {
        n_in = gi.graphInfoV3.numGraphInputs;  gin = gi.graphInfoV3.graphInputs;
        n_out = gi.graphInfoV3.numGraphOutputs; gout = gi.graphInfoV3.graphOutputs;
        chosen_name = gi.graphInfoV3.graphName;
    }
    std::string chosen_graph_name = chosen_name ? chosen_name : "";

    // Deep-copy tensor descriptors and allocate native client buffers.
    m->inputs.resize(n_in);
    m->outputs.resize(n_out);
    for (uint32_t i = 0; i < n_in; ++i) {
        m->inputs[i].version = QNN_TENSOR_VERSION_1;
        m->inputs[i].v1 = QNN_TENSOR_V1_INIT;
        deep_copy_tensor(m->inputs[i], gin[i]);
        size_t elems = 1; for (uint32_t d = 0; d < tensor_rank(m->inputs[i]); ++d) elems *= tensor_dims(m->inputs[i])[d];
        size_t nbytes = elems * dtype_size_bytes(tensor_dtype(m->inputs[i]));
        void* p = calloc(1, nbytes);
        tensor_set_membuf(m->inputs[i], p, (uint32_t)nbytes);
        m->input_infos.push_back(make_info(m->inputs[i]));
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        m->outputs[i].version = QNN_TENSOR_VERSION_1;
        m->outputs[i].v1 = QNN_TENSOR_V1_INIT;
        deep_copy_tensor(m->outputs[i], gout[i]);
        size_t elems = 1; for (uint32_t d = 0; d < tensor_rank(m->outputs[i]); ++d) elems *= tensor_dims(m->outputs[i])[d];
        size_t nbytes = elems * dtype_size_bytes(tensor_dtype(m->outputs[i]));
        void* p = calloc(1, nbytes);
        tensor_set_membuf(m->outputs[i], p, (uint32_t)nbytes);
        m->output_infos.push_back(make_info(m->outputs[i]));
    }

    m->sys_iface.systemContextFree(sys_ctx);

    // 4. backend + device + context-from-binary + graph retrieve
    if (m->iface.backendCreate(nullptr, nullptr, &m->backend) != QNN_SUCCESS) {
        SET_ERR(m, "backendCreate failed"); return m;
    }
    if (m->iface.deviceCreate) {
        Qnn_ErrorHandle_t rc = m->iface.deviceCreate(nullptr, nullptr, &m->device);
        if (rc != QNN_SUCCESS && rc != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
            SET_ERR(m, "deviceCreate failed"); return m;
        }
    }
    if (m->iface.contextCreateFromBinary(m->backend, m->device, nullptr,
                                          buf.data(), buf.size(),
                                          &m->context, nullptr) != QNN_SUCCESS) {
        SET_ERR(m, "contextCreateFromBinary failed"); return m;
    }
    if (m->iface.graphRetrieve(m->context, chosen_graph_name.c_str(), &m->graph) != QNN_SUCCESS) {
        SET_ERR(m, "graphRetrieve failed"); return m;
    }

    m->last_error = "ok";
    return m;
}

int32_t qnn_execute(qnn_model_t* m, const float* const* inputs, float* const* outputs) {
    if (!m || !m->graph) { if (m) SET_ERR(m, "execute on null/invalid model"); return -1; }

    // Quantize floats into each input client buffer.
    for (size_t i = 0; i < m->inputs.size(); ++i) {
        const auto& info = m->input_infos[i];
        Qnn_DataType_t dt = (Qnn_DataType_t)info.qnn_dtype;
        void* dst = nullptr;
        uint32_t dst_bytes = 0;
        if (m->inputs[i].version == QNN_TENSOR_VERSION_2) {
            dst = m->inputs[i].v2.clientBuf.data;
            dst_bytes = m->inputs[i].v2.clientBuf.dataSize;
        } else {
            dst = m->inputs[i].v1.clientBuf.data;
            dst_bytes = m->inputs[i].v1.clientBuf.dataSize;
        }
        const float* src = inputs[i];
        size_t n = info.num_elements;
        if (dt == QNN_DATATYPE_UFIXED_POINT_8) {
            uint8_t* d = (uint8_t*)dst;
            const float s = info.scale;
            const int   o = info.offset;
            for (size_t k = 0; k < n; ++k) {
                float q = std::round(src[k] / s) - (float)o;
                if (q < 0)   q = 0;
                if (q > 255) q = 255;
                d[k] = (uint8_t)q;
            }
        } else if (dt == QNN_DATATYPE_SFIXED_POINT_8) {
            int8_t* d = (int8_t*)dst;
            const float s = info.scale;
            const int   o = info.offset;
            for (size_t k = 0; k < n; ++k) {
                float q = std::round(src[k] / s) - (float)o;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                d[k] = (int8_t)q;
            }
        } else if (dt == QNN_DATATYPE_FLOAT_32) {
            memcpy(dst, src, dst_bytes);
        } else {
            SET_ERR(m, "unsupported input dtype (only UF8/SF8/FP32 implemented)");
            return -2;
        }
    }

    // Execute graph
    Qnn_ErrorHandle_t rc = m->iface.graphExecute(m->graph,
                                                  m->inputs.data(),  (uint32_t)m->inputs.size(),
                                                  m->outputs.data(), (uint32_t)m->outputs.size(),
                                                  nullptr, nullptr);
    if (rc != QNN_GRAPH_NO_ERROR) {
        char ebuf[128];
        snprintf(ebuf, sizeof(ebuf), "graphExecute failed: 0x%llx", (unsigned long long)rc);
        SET_ERR(m, ebuf);
        return -3;
    }

    // Dequantize each output buffer to float.
    for (size_t i = 0; i < m->outputs.size(); ++i) {
        const auto& info = m->output_infos[i];
        Qnn_DataType_t dt = (Qnn_DataType_t)info.qnn_dtype;
        const void* src = nullptr;
        if (m->outputs[i].version == QNN_TENSOR_VERSION_2)
            src = m->outputs[i].v2.clientBuf.data;
        else
            src = m->outputs[i].v1.clientBuf.data;
        float* d = outputs[i];
        size_t n = info.num_elements;
        if (dt == QNN_DATATYPE_UFIXED_POINT_8) {
            const uint8_t* s = (const uint8_t*)src;
            const float scale = info.scale;
            const int   off   = info.offset;
            for (size_t k = 0; k < n; ++k) d[k] = ((float)s[k] + (float)off) * scale;
        } else if (dt == QNN_DATATYPE_SFIXED_POINT_8) {
            const int8_t* s = (const int8_t*)src;
            const float scale = info.scale;
            const int   off   = info.offset;
            for (size_t k = 0; k < n; ++k) d[k] = ((float)s[k] + (float)off) * scale;
        } else if (dt == QNN_DATATYPE_FLOAT_32) {
            memcpy(d, src, n * sizeof(float));
        } else {
            SET_ERR(m, "unsupported output dtype (only UF8/SF8/FP32 implemented)");
            return -4;
        }
    }

    return 0;
}

} // extern "C"
