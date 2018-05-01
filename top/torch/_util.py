""" PyTorch Utilitaries """
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['save_csv']

def save_csv(tensor, filename, **kwargs):
    """ Save a tensor as a csv file.

    Args:
        tensor (torch.Tensor): Tensor you want to save
        filename (string): Filename for the csv file
        kwargs (kwargs, optional): Extra keyword arguments to pass on to the :fun:`numpy.savetxt` function
    """
    np.savetxt(filename, tensor.numpy(), delimiter=',', **kwargs)
