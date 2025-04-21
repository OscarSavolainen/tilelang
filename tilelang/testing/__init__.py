# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import sys
import inspect
import pytest
import random
import torch
import numpy as np
from tvm.testing.utils import *

from tilelang.utils.tensor import torch_assert_close as torch_assert_close
from tvm.contrib.rocm import get_rocm_arch, find_rocm_path


# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def requires_rocm_architecture(required_archs: tuple) -> str:
    rocm_path = find_rocm_path()
    arch = get_rocm_arch(rocm_path)
    import ipdb pprint; ipdb.set_trace();
    if arch in required_archs:
        return True
