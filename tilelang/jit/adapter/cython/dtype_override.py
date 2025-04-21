import torch
from .adapter import CythonKernelAdapter

def hip_fp8_fnuz_override(artifact, tilelang_func):
    """
    For the Cython kernel adaproir, prior to compilation, we override the use of the FP8 e4m3/e5m2 
    OCP dtypes with AMD-supported de4m3/e5m2 fnuz dypes, enabling us to do FP8 matmuls.
    """
    # Overwrite expected params to chnage dtype to AMD-supported fnuz type. This is because
    # TVM does not yet have support for fp8 fnuz dtype.
    for index, param in enumerate(artifact.params):
        if param.dtype == torch.float8_e4m3fn:
            artifact.params[index].dtype = torch.float8_e4m3fnuz
        if param.dtype == torch.float8_e5m2:
            artifact.params[index].dtype = torch.float8_e5m2fnuz

    # We get the default buffer dtype map based off of the TVM returned function, and 
    # overwrite the fp8 param dtypes
    buffer_dtype_map_base = CythonKernelAdapter.process_buffer_dtype(tilelang_func)
    buffer_dtype_map = {}
    for param, value in buffer_dtype_map_base.items():
        if value[1] == torch.float8_e4m3fn:
            buffer_dtype_map[param] = (value[0], torch.float8_e4m3fnuz)
        if value[1] == torch.float8_e5m2:
            buffer_dtype_map[param] = (value[0], torch.float8_e5m2fnuz)

    return artifact, buffer_dtype_map
