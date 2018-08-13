""" Visualisation of PyTorch stuff """
import math
import logging
import functools
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

__all__ = ['show_tensor', 'show_tensor_freq', 'show_weights', 'get_activations']
log = logging.getLogger(__name__)


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


def show_tensor_freq(tensor, dim=None, bins=None, ax=plt, **kwargs):
    """ Plot the frequencies of a tensor in a matplotlib histogram chart.

    Args:
        tensor (torch.Tensor): Tensor to analyze
        dim (int, optional): Dimension to compare the tensor on; Default **Don't compare, but rather plot hist of all values**
        bins (optional): bins for the histogram. This value gets passed on to :func:`matplotlib.pyplot.hist`; Default **None**
        ax (matlpotlib.axes.Axes, optional): Axes to plot the histogram in; Default **matplotlib.pyplot**
        kwargs (optional): Extra arguments to pass on to the :func:`matplotlib.pyplot.hist` funciotn; Default **None**

    Returns:
        (tuple): Return values from the :func:`matplotlib.pyplot.hist` function.
    """
    if dim is not None and dim >= tensor.dim():
        raise ValueError('dim variable is bigger than the number of dimensions of the tensor [{dim}/{tensor.dim()}]')

    if dim is not None:
        size = tensor.size(dim)
        if dim != tensor.dim()-1:
            tensor = tensor.transpose(tensor.dim()-1, dim).contiguous()
        tensor = tensor.view(-1, size).cpu().numpy()
        label = [f'c{n}' for n in range(size)]
    else:
        tensor = tensor.view(-1).cpu().numpy()
        label = 'tensor'

    return ax.hist(tensor, bins, label=label, **kwargs)


def show_weights(weights, normalize=False, orientation='portrait'):
    """ Visualize the weights of a layer in a matplotlib plot

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


def get_activations(modules, return_output=True, save_folder=None, grid=False, normalize=False, border_fill=255):
    """ Save activation maps of underlying layer operations by registering a forward hook on the different modules.
    This function can save the feature maps as an n x n image grid and can also return the feature maps as numpy arrays.

    Args:
        modules (iterable): The modules for which you want to save the output feature map 
        return_output (Boolean, optional): Save the feature maps in a dictionary that will be returned from this function; Default **True**
        save_folder (str, optional): Folder to save the feature maps as images; Default **Do not save images**
        grid (Boolean, optional): Stack the different feature maps in a square grid; Default **False**
        normalize (Boolean, optional): Normalize the feature maps between 0-255; Default **False**
        border_fill (int, optional): Fill value between the feature maps when constructing the grid; Default **255**

    Returns:
        (tuple): [list of hook_handles], output_dict

    Warning:
        The returned output dict will be empty at first, but will fill only up once you run the network!

    Note:
        This function will only save the feature maps of the first item in the batch.

    Note:
        The `modules` iterable can return **name, modules** tuples as well.
        In that case the name will be used as key for the output dict and in the filename of the saved images.
        Otherwise an increasing number is used.

    Note:
        When saving the feature maps as images, they will be always normalized and saved as a grid.
        This will not impact your argument flags, when returning the maps as an output as well!
    """
    def save_activations(name, mod, inp, out):
        if out.dim() != 4:
            log.error(f'Output tensor of {name} does not have 4 dimensions [{out.shape}]')
            return
    
        activation = out.detach().cpu().clone().numpy()[0]
        
        # Normalize to 0-255
        if normalize:
            min_val = activation.min()
            activation -= min_val
            max_val = activation.max()
            activation *= 255 / max_val
            activation = activation.astype(np.uint8)
    
        # Create grid of tensor
        if grid or save_folder is not None:
            if not normalize:
                min_val = activation.min()
                activation -= min_val
                max_val = activation.max()
                tmp_act = (activation * (255 / max_val)).astype(np.uint8)
            else:
                tmp_act = activation

            grid_size = math.ceil(math.sqrt(tmp_act.shape[0]))
            h, w = tmp_act.shape[1:]
            img = np.ones([grid_size, grid_size, h+2, w+2], dtype=np.uint8) * border_fill
    
            for i in range(tmp_act.shape[0]):
                x_grid = i % grid_size
                y_grid = i // grid_size
                img[y_grid, x_grid, 1:h+1, 1:w+1] = tmp_act[i]
            img = np.reshape(img.transpose((0,2,1,3)), [grid_size*(h+2), grid_size*(w+2)])

            if grid:
                activation = img
    
        # Save activation
        if save_folder is not None:
            path = Path(save_folder) / (name + '.png')
            img = Image.fromarray(img)
            img.save(str(path))

        if output is not None:
            output[name] = activation

    if return_output:
        output = {}
    else:
        output = None

    if save_folder is not None:
        f = Path(save_folder)
        f.mkdir(parents=True, exist_ok=True)

    ret_handles = []
    for i, mod in enumerate(modules):
        if isinstance(mod, tuple):
            name, mod = mod
        else:
            name = str(i)

        ret_handles.append(mod.register_forward_hook(functools.partial(save_activations, name)))

    return ret_handles, output
