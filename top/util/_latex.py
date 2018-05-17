"""LaTeX stuff for publications"""
import logging
from math import sqrt
import matplotlib.pyplot as plt

__all__ = ['mpl_latex', 'mpl_latex_axes', 'latex_table']
log = logging.getLogger(__name__)


def mpl_latex(fig_width=None, fig_height=None, columns=1):
    """ Set up matplotlib's RC params for LaTeX publication ready plots.
    Code stolen and adapted from: https://nipunbatra.github.io/blog/2014/latexify.html

    Args:
        fig_width (float, optional): Figure width in inches
        fig_height (float, optional): Figure height in inches
        columns ({1,2}, optional): Whether the image is to be used in 1 or 2 columns (when no dims are given)

    Note:
        Use ``plt.savefig()`` in stead of ``plt.show()`` after using this figure!
    """
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        log.warn(f'Fig_height too large [{fig_height}], reducing to {MAX_HEIGHT_INCHES} inches.')
        fig_height = MAX_HEIGHT_INCHES

    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': True,
        'figure.figsize': [fig_width,fig_height],
        'font.family': 'serif',
        'font.serif': ['Times', 'Computer Modern Roman'],
    }

    matplotlib.rcParams.update(params)


def mpl_latex_axes(ax):
    """ Bettter axes for publications.
    Code stolen and adapted from: https://nipunbatra.github.io/blog/2014/latexify.html

    Args:
        ax (matplotlib.axes.Axes): Axes to format
    """
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out')

    return ax


def latex_table(data, out_file=None, alignment=None, header=None):
    """ Creates a LaTeX tabular table from your data.

    Args:
        data (2D list or pandas.DataFrame): data for table
        out_file (str, optional): Output filename; Default **no file created**
        alignment (str, optional): Tabular alignment string; Default **left align all columns with lines in between**
        header (list, optional): Header to be inserted before data; Default **Bold DataFrame column names or nothing**

    Returns:
        str: LaTeX tabular table as a list of strings

    Example:
       You should include the resulting .tex file in the following way

       .. highlight:: tex
          
          \\begin{table}
            \\centering
            \\input{generated_latex_file.tex}
            \\caption{Example of a generated tex table}
            \\label{tab:generated_table}
          \\end{table}
    """
    # Parse Arguments
    if alignment is None:
        if isinstance(data, list):
            alignment = '|' + 'l|' * len(data[0])
        else:
            alignment = '|' + 'l|' * len(data.columns)

    if header is None and not isinstance(data, list):
        header = list(data.columns)
        header = ['\\textbf{'+head+'}' for head in header]

    # Create table
    latex_str = ['\\begin{tabular}{'+alignment+'}']
    if header is not None:
        latex_str.append(' & '.join(header) + ' \\\\')
        latex_str.append('\hline')

    if isinstance(data, list):
        for d in data:
            latex_str.append(' & '.join([str(item) for item in d]) + ' \\\\')
    else:
        for row in data.iterrows():
            d = list(row[1])
            latex_str.append(' & '.join([str(item) for item in d]) + ' \\\\')

    latex_str.append('\\end{tabular}')

    # Write file
    if out_file is not None:
        with open(out_file, 'w') as fp:
            fp.write('\n'.join(latex_str))

    return latex_str
