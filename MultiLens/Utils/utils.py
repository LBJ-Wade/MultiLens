__author__ = 'sibirrer'

import numpy as np

import MultiLens.Utils.constants as const


def make_grid(numPix, deltapix):
    """
    returns x, y position information in two 1d arrays
    """
    a = np.arange(numPix)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x_grid = (matrix[:, 0] - numPix/2.)*deltapix
    y_grid = (matrix[:, 1] - numPix/2.)*deltapix
    return x_grid*const.arcsec, y_grid*const.arcsec


def array2image(array):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    n=int(np.sqrt(len(array)))
    if n**2 != len(array):
        raise ValueError("lenght of input array given as %s is not square of integer number!" %(len(array)))
    image = array.reshape(n, n)
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh