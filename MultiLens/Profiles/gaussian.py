__author__ = 'sibirrer'

#this file contains a class to make a gaussian

import numpy as np

class Gaussian(object):
    """
    this class contains functions to evaluate a Gaussian function and calculates its derivative and hessian matrix
    """
    def function(self, x, y, amp, sigma_x, sigma_y, center_x, center_y):
        """
        returns Gaussian
        """
        sigma_x = float(sigma_x)
        sigma_y = float(sigma_y)
        return amp/(2*np.pi*sigma_x*sigma_y) *np.exp(-(((center_x-x)/sigma_x)**2+((center_y-y)/sigma_y)**2)/2)

    def derivatives(self, x, y, amp, sigma_x, sigma_y, center_x = 0, center_y = 0):
        """
        returns df/dx and df/dy of the function
        """
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        return f_ * (center_x-x)/sigma_x**2, f_ * (center_y-y)/sigma_y**2

    def hessian(self, x, y, amp, sigma_x, sigma_y, center_x = 0, center_y = 0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        f_xx = f_ * ( (-1./sigma_x**2) + (center_x-x)**2/sigma_x**4 )
        f_yy = f_ * ( (-1./sigma_y**2) + (center_y-y)**2/sigma_y**4 )
        f_xy = f_ * (center_x-x)/sigma_x**2 * (center_y-y)/sigma_y**2
        return f_xx, f_yy, f_xy

    def all(self, x, y, amp, sigma_x, sigma_y, center_x = 0, center_y = 0):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        f_x = f_ * (center_x-x)/sigma_x**2
        f_y = f_ * (center_y-y)/sigma_y**2
        f_xx = f_ * ( (-1./sigma_x**2) + (center_x-x)**2/sigma_x**4 )
        f_yy = f_ * ( (-1./sigma_y**2) + (center_y-y)**2/sigma_y**4 )
        f_xy = f_ * (center_x-x)/sigma_x**2 * (center_y-y)/sigma_y**2
        return f_, f_x, f_y, f_xx, f_yy, f_xy