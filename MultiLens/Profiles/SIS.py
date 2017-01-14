__author__ = 'sibirrer'

import numpy as np
import MultiLens.Utils.constants as const

class SIS(object):
    """
    this class contains the function and the derivatives of the Singular Isothermal Sphere in Physical coordinates
    """
    def function(self, x, y, sigma_v, pos_x=0, pos_y=0):
        x_shift = x - pos_x
        y_shift = y - pos_y
        f_ = 2*(sigma_v/const.c)**2 * np.sqrt(x_shift*x_shift + y_shift*y_shift)
        #TODO not right dimensions!!!
        return f_

    def derivative(self, x, y, sigma_v, pos_x=0, pos_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_shift = x - pos_x
        y_shift = y - pos_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        phi = 4*np.pi*(sigma_v/const.c)**2
        if isinstance(R, int) or isinstance(R, float):
            a = phi/max(0.000001, R)
        else:
            a=np.empty_like(R)
            r = R[R > 0]  #in the SIS regime
            a[R == 0] = 0
            a[R > 0] = phi/r
        f_x = a * x_shift
        f_y = a * y_shift
        return f_x, f_y

    def hessian(self, x, y, sigma_v, pos_x=0, pos_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_shift = x - pos_x
        y_shift = y - pos_y
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)
        phi = 4*np.pi*(sigma_v/const.c)**2
        if isinstance(R, int) or isinstance(R, float):
            prefac = phi/max(0.000001, R)
        else:
            prefac = np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            prefac[R==0] = 0.
            prefac[R>0] = phi/r

        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_xx, f_yy, f_xy

    def all(self, x, y, sigma_v, pos_x=0, pos_y=0):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        x_shift = x - pos_x
        y_shift = y - pos_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        phi = 4*np.pi*(sigma_v/const.c)**2
        if isinstance(R, int) or isinstance(R, float):
            a = phi/max(0.000001,R)
        else:
            a=np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            a[R==0] = 0.
            a[R>0] = phi/r

        f_ = 2*(sigma_v/const.c)**2 * R
        f_x = a * x_shift
        f_y = a * y_shift
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)

        if isinstance(R, int) or isinstance(R, float):
            prefac = phi/max(0.000001, R)
        else:
            prefac = np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            prefac[R==0] = 0.
            prefac[R>0] = phi/r

        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_, f_x, f_y, f_xx, f_yy, f_xy
