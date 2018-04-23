""" Visualisation of PyTorch stuff """
import torch
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['show_tensor', 'show_weights']

def show_tensor(tensor, ax=plt):
    """ Show a tensor in a matplotlib window.
        
    Args:
        tensor (torch.Tensor): Tensor to show
        ax (matlpotlib.axes.Axes, optional): Axes to show the tensor in; Default **matplotlib.pyplot**

    Note:
        The tensor should be of dimensions:
            - M x N
            - 3 x M x N
            - 4 x M x N
    """
    if tensor.dim() != 3 and tensor.dim() != 2:
        raise TypeError('Tensor needs to have 2 or 3 dimensions')

    if tensor.dim() == 3 and (tensor.size(0) != 3 and tensor.size(0) != 4):
        raise TypeError('Tensor needs to be of size [3 x M x N] or [4 x M x N]')

    tensor = tensor.cpu().numpy()
    if tensor.ndim == 3:
        np.rollaxis(tensor, 0, 2)

    ax.imshow(tensor, cmap='gray')

def show_weights(weights, normalize=False):
    """ Visualize the weights of a layer

    Args:
        weights (torch.Tensor): Tensor containing the weights you want to visualize
        normalize (Boolean, optional): Whether or not to normalize the weights between 0-1; Default **False**
        orientation (str, optional): 'landscape' or 'portrait' for the organisation of the subplots; Default **'portrait'**

    Returns:
        (matplotlib.figure.Figure): Figure with the weights drawn
    """
    if weights.dim() != 4:
        raise TypeError('Can only visualize weights with 4 dimensions')

    out_chan = weights.size(0)
    in_chan = weights.size(1)

    if orientation == 'landscape':
        if out_chan > in_chan:
            fig, axes = plt.subplots(in_chan, out_chan)
            axes = np.array(axes).reshape(in_chan, out_chan).transpose()
            orientation = True
        else:
            fig, axes = plt.subplots(out_chan, in_chan)
            axes = np.array(axes).reshape(out_chan, in_chan)
            orientation = False
    else:
        if in_chan > out_chan:
            fig, axes = plt.subplots(in_chan, out_chan)
            axes = np.array(axes).reshape(in_chan, out_chan).transpose()
            orientation = True
        else:
            fig, axes = plt.subplots(out_chan, in_chan)
            axes = np.array(axes).reshape(out_chan, in_chan)
            orientation = False

    if normalize:
        max_num = weights.max()
        min_num = weights.min()
        weights = (weights - min_num) * (1 / (max_num - min_num))

    for i in range(out_chan):
        if orientation:
            axes[i][0].get_xaxis().set_label_position('top')
            axes[i][0].set_xlabel(f'out {i}')
        else:
            axes[i][0].set_ylabel(f'out {i}')
        for j in range(in_chan):
            if i == 0:
                if orientation:
                    axes[0][j].set_ylabel(f'in {j}')
                else:
                    axes[0][j].get_xaxis().set_label_position('top')
                    axes[0][j].set_xlabel(f'in {j}')

            axes[i][j].get_xaxis().set_ticks([])
            axes[i][j].get_xaxis().set_ticklabels([])
            axes[i][j].get_yaxis().set_ticks([])
            axes[i][j].get_yaxis().set_ticklabels([])

            show_tensor(weights[i][j], axes[i][j])

    return fig
