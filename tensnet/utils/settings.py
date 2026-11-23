from . import config


def set_arch(arch):
    if arch in ["stacked", "sequential", "xyrev", "xrevy"]:
        config.__ARCH_2D = arch
    else:
        raise ValueError(f"{arch} not in [stacked,sequential,xyrev,xrevy]")


def set_comp_method(method):
    if method in ["randomized", "svd", "randomized_svd", "off"]:
        config.__compress_method = method
    else:
        raise ValueError(f"{method} not supported, must be one of randomized or svd or off.")


def set_accuracy(accuracy):
    config.__COMP_ACC = accuracy


def set_bd_limit(limit: int):
    config.__BD_LIMIT = limit


def set_param(key, value):
    config.__parameters[str(key)] = value


def set_device(dev):
    config.__device = dev
