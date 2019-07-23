from matplotlib import cm
from matplotlib.colors import rgb2hex
import numpy as np
import pandas as pd
    
def background_gradient(self, cvals=None, cmin=None, cmax=None, cmap='viridis'):
    """For use with `DataFrame.style.apply` this function will apply a heatmap
    color gradient *elementwise* to the calling DataFrame

    Parameters
    ----------
    self : pd.DataFrame
        The calling DataFrame. This argument is automatically passed in by the
        `DataFrame.style.apply` method

    cvals : pd.DataFrame
        If specified this DataFrame is used to determine the color gradient

    cmin : float
        If specified, any values below this will be clipped to the bottom of
        the cmap

    cmax : float
        If specified, any values above this will be clipped to the top of
        the cmap

    cmap : colormap or str
        The colormap to use

    Returns
    -------
    pd.DataFrame
        The css styles to apply

    """

    if cvals is None:
        cvals = self.values.ravel().copy()
    else:
        assert cvals.shape == self.shape
        cvals = cvals.values.ravel().copy()
    cvals -= cmin #or cvals.min()
    cvals /= (cmax - cmin) #or cvals.max()

    styles = []
    for rgb in cmap(cvals): #cm.viridis_r(cvals):
        style = []
        style.append("background-color: {}".format(rgb2hex(rgb)))
        styles.append('; '.join(style))
    styles = np.asarray(styles).reshape(self.shape)
    return pd.DataFrame(styles, index=self.index, columns=self.columns)