__COMP_ACC = 1e-12
__BD_LIMIT = None
__ARCH_2D = "stacked"
__parameters = {"a": 0.0, "b": 2.0, "L": 8, "disable_compile": True, "low_acc": False}
__device = "cpu"
__compress_method = "randomized"


def COMP_ACC():
    return __COMP_ACC


def BD_LIMIT() -> int:
    return __BD_LIMIT


def device():
    return __device


def ARCH_2D():
    return __ARCH_2D


def COMPRESS_METHOD():
    return __compress_method


def parameters(key=None):
    if key in ["ax", "bx", "ay", "by", "Lx", "Ly"] and key not in __parameters.keys():
        return __parameters[key[0]]
    elif key is not None:
        return __parameters[key]
    return __parameters
