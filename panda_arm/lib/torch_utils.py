import numpy as np
import torch


def to_torch(x, device='cpu'):

    if type(x) == np.ndarray:
        # Convert plain arrays
        out = torch.from_numpy(x).float().to(device)
    elif type(x) == dict:
        # Convert dictionaries of values
        out = {}
        for key, value in x.items():
            out[key] = to_torch(value, device)
    elif type(x) in (list, tuple):
        # Convert lists of values
        out = []
        for value in x:
            out.append(to_torch(value, device))
    else:
        assert False, "Invalid datatype {}!".format(type(x))

    return out


def to_numpy(x):
    if type(x) == torch.Tensor:
        # Convert plain tensors
        out = x.detach().cpu().numpy()
    elif type(x) == dict:
        # Convert dictionaries of values
        out = {}
        for key, value in x.items():
            out[key] = from_torch(value)
    elif type(x) in (list, tuple):
        # Convert lists of values
        out = []
        for value in x:
            out.append(from_torch(value))
    else:
        assert False, "Invalid datatype {}!".format(type(x))

    return out
