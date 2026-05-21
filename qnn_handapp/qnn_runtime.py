"""ctypes wrapper around libqnn_shim.so.

Each :class:`QnnModel` owns a QNN context bound to one graph from a context
binary. Inputs and outputs are exchanged as float32 numpy arrays; the C
shim handles per-tensor quantization to/from UFIXED_POINT_8.
"""
from __future__ import annotations

import ctypes
import os
from typing import List, Sequence

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SO = os.path.join(_HERE, "libqnn_shim.so")


class _TensorInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("rank", ctypes.c_int32),
        ("dims", ctypes.c_int32 * 8),
        ("qnn_dtype", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("offset", ctypes.c_int32),
        ("num_elements", ctypes.c_size_t),
    ]


class TensorMeta:
    __slots__ = ("name", "shape", "dtype_id", "scale", "offset", "num_elements")

    def __init__(self, raw: _TensorInfo):
        self.name = raw.name.decode("utf-8", errors="replace")
        self.shape = tuple(raw.dims[i] for i in range(raw.rank))
        self.dtype_id = int(raw.qnn_dtype)
        self.scale = float(raw.scale)
        self.offset = int(raw.offset)
        self.num_elements = int(raw.num_elements)

    def __repr__(self) -> str:
        return (
            f"TensorMeta(name={self.name!r} shape={self.shape} "
            f"dtype_id={self.dtype_id} scale={self.scale} offset={self.offset})"
        )


def _load(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    lib.qnn_load.restype = ctypes.c_void_p
    lib.qnn_load.argtypes = [ctypes.c_char_p] * 4
    lib.qnn_destroy.argtypes = [ctypes.c_void_p]
    lib.qnn_last_error.restype = ctypes.c_char_p
    lib.qnn_last_error.argtypes = [ctypes.c_void_p]
    lib.qnn_num_inputs.restype = ctypes.c_int32
    lib.qnn_num_outputs.restype = ctypes.c_int32
    lib.qnn_num_inputs.argtypes = [ctypes.c_void_p]
    lib.qnn_num_outputs.argtypes = [ctypes.c_void_p]
    lib.qnn_input_info.restype = ctypes.POINTER(_TensorInfo)
    lib.qnn_output_info.restype = ctypes.POINTER(_TensorInfo)
    lib.qnn_input_info.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qnn_output_info.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qnn_execute.restype = ctypes.c_int32
    lib.qnn_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    return lib


class QnnRuntime:
    """Holds a loaded libqnn_shim.so plus device-side library paths."""

    def __init__(self, backend_so: str, system_so: str, shim_so: str = _DEFAULT_SO):
        self.backend_so = backend_so.encode("utf-8")
        self.system_so = system_so.encode("utf-8")
        self.lib = _load(shim_so)


class QnnModel:
    def __init__(self, runtime: QnnRuntime, binary_path: str, graph_name: str | None = None):
        gn = graph_name.encode("utf-8") if graph_name else None
        self._lib = runtime.lib
        handle = self._lib.qnn_load(runtime.backend_so, runtime.system_so,
                                    binary_path.encode("utf-8"), gn)
        if not handle:
            raise RuntimeError("qnn_load returned NULL")
        self._h = ctypes.c_void_p(handle)

        # Inspect error even on success path because shim writes "ok" on success.
        err = self._lib.qnn_last_error(self._h).decode("utf-8", errors="replace")
        if err != "ok":
            self._lib.qnn_destroy(self._h)
            raise RuntimeError(f"qnn_load failed: {err}")

        n_in = self._lib.qnn_num_inputs(self._h)
        n_out = self._lib.qnn_num_outputs(self._h)
        self.inputs: List[TensorMeta] = [
            TensorMeta(self._lib.qnn_input_info(self._h, i).contents) for i in range(n_in)
        ]
        self.outputs: List[TensorMeta] = [
            TensorMeta(self._lib.qnn_output_info(self._h, i).contents) for i in range(n_out)
        ]

        # Pre-allocate output buffers shared across calls.
        self._out_bufs = [
            np.zeros(t.num_elements, dtype=np.float32) for t in self.outputs
        ]

        # ctypes arrays of pointers reused per call.
        self._in_ptrs = (ctypes.POINTER(ctypes.c_float) * n_in)()
        self._out_ptrs = (ctypes.POINTER(ctypes.c_float) * n_out)()
        for i, buf in enumerate(self._out_bufs):
            self._out_ptrs[i] = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def execute(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        if len(inputs) != len(self.inputs):
            raise ValueError(f"expected {len(self.inputs)} inputs, got {len(inputs)}")
        in_contig: List[np.ndarray] = []
        for i, (arr, meta) in enumerate(zip(inputs, self.inputs)):
            a = np.ascontiguousarray(arr, dtype=np.float32)
            if a.size != meta.num_elements:
                raise ValueError(
                    f"input {i} ({meta.name}) expects {meta.num_elements} "
                    f"floats, got {a.size}"
                )
            in_contig.append(a)
            self._in_ptrs[i] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._lib.qnn_execute(self._h, self._in_ptrs, self._out_ptrs)
        if rc != 0:
            err = self._lib.qnn_last_error(self._h).decode("utf-8", errors="replace")
            raise RuntimeError(f"qnn_execute rc={rc}: {err}")
        # Reshape and return *copies* so callers can hold them across runs.
        results = []
        for buf, meta in zip(self._out_bufs, self.outputs):
            results.append(buf.reshape(meta.shape).copy())
        return results

    def close(self):
        if getattr(self, "_h", None):
            self._lib.qnn_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
