import torch


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    # return torch.log(torch.clamp(x, min=clip_val))*C
    return torch.log(x+1)*C  # Winner being used for newest and distributed
    # return x/60000


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    # return torch.exp(x/C)
    return torch.exp(x/C)-1  # Winner being used for newest and distributed
    # return x*60000
