import numpy as np
from torch.autograd import Variable


def to_text(x, dataset):
    """Convert iterable of indices to text"""
    return "".join(map(dataset.int_to_char_map.get, x))


def new_hidden(model, batch_size):
    return repackage_hidden(model.init_hidden(batch_size))


def repackage_sample(x, config):
    """Helper for converting to a Variable"""
    x = Variable(x)
    if config.cuda:
        x = x.cuda()
    return x


def repackage_hidden(h):
    """
    Repackages a hidden state as a new variable. The stops gradients from
    flowing back further.
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_sampling_function(config):
    if config.sampling_mode == "argmax":
        return np.argmax
    elif config.sampling_mode == "weighted":
        return weighted_pick
    else:
        raise KeyError(config.sampling_mode)


def weighted_pick(weights):
    """Sample from a distribution defined by weights. Returns index"""
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1)*s))
