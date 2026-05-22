"""ctypes wrapper around libsnpe_shim.so.

A :class:`SnpeModel` owns an in-process SNPE instance bound to one DLC and runs
it on HTP (or CPU). Inputs and outputs are exchanged as float32 numpy arrays;
SNPE quantizes to/from the DLC's fixed-point encoding internally.
"""
from __future__ import annotations

import ctypes
import os
from typing import List, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SO = os.path.join(_HERE, "libsnpe_shim.so")


class _TensorInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("rank", ctypes.c_int32),
        ("dims", ctypes.c_int32 * 8),
        ("num_elements", ctypes.c_size_t),
    ]


class TensorMeta:
    __slots__ = ("name", "shape", "num_elements")

    def __init__(self, raw: _TensorInfo):
        self.name = raw.name.decode("utf-8", errors="replace")
        self.shape = tuple(raw.dims[i] for i in range(raw.rank))
        self.num_elements = int(raw.num_elements)

    def __repr__(self) -> str:
        return f"TensorMeta(name={self.name!r} shape={self.shape})"


def _load(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    lib.snpe_load.restype = ctypes.c_void_p
    lib.snpe_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    lib.snpe_destroy.argtypes = [ctypes.c_void_p]
    lib.snpe_last_error.restype = ctypes.c_char_p
    lib.snpe_last_error.argtypes = [ctypes.c_void_p]
    lib.snpe_num_inputs.restype = ctypes.c_int32
    lib.snpe_num_outputs.restype = ctypes.c_int32
    lib.snpe_num_inputs.argtypes = [ctypes.c_void_p]
    lib.snpe_num_outputs.argtypes = [ctypes.c_void_p]
    lib.snpe_input_info.restype = ctypes.POINTER(_TensorInfo)
    lib.snpe_output_info.restype = ctypes.POINTER(_TensorInfo)
    lib.snpe_input_info.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.snpe_output_info.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.snpe_execute.restype = ctypes.c_int32
    lib.snpe_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    return lib


class SnpeModel:
    """In-process SNPE instance for a single DLC."""

    def __init__(self, dlc_path: str, use_dsp: bool = True,
                 accelerated_init: bool = True, shim_so: str = _DEFAULT_SO,
                 output_names: Sequence[str] | None = None):
        self._lib = _load(shim_so)
        out = ",".join(output_names).encode("utf-8") if output_names else None
        handle = self._lib.snpe_load(
            dlc_path.encode("utf-8"), int(use_dsp), int(accelerated_init), out)
        if not handle:
            raise RuntimeError("snpe_load returned NULL")
        self._h = ctypes.c_void_p(handle)

        err = self._lib.snpe_last_error(self._h).decode("utf-8", errors="replace")
        if err != "ok":
            self._lib.snpe_destroy(self._h)
            self._h = None
            raise RuntimeError(f"snpe_load failed: {err}")

        n_in = self._lib.snpe_num_inputs(self._h)
        n_out = self._lib.snpe_num_outputs(self._h)
        self.inputs: List[TensorMeta] = [
            TensorMeta(self._lib.snpe_input_info(self._h, i).contents) for i in range(n_in)
        ]
        self.outputs: List[TensorMeta] = [
            TensorMeta(self._lib.snpe_output_info(self._h, i).contents) for i in range(n_out)
        ]

        self._out_bufs = [np.zeros(t.num_elements, dtype=np.float32) for t in self.outputs]
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
                    f"floats, got {a.size}")
            in_contig.append(a)
            self._in_ptrs[i] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._lib.snpe_execute(self._h, self._in_ptrs, self._out_ptrs)
        if rc != 0:
            err = self._lib.snpe_last_error(self._h).decode("utf-8", errors="replace")
            raise RuntimeError(f"snpe_execute rc={rc}: {err}")
        return [buf.reshape(meta.shape).copy()
                for buf, meta in zip(self._out_bufs, self.outputs)]

    def close(self):
        if getattr(self, "_h", None):
            self._lib.snpe_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
