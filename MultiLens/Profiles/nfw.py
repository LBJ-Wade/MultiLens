__author__ = 'sibirrer'


import numpy as np

from MultiLens.Utils.halo_param import HaloParam
import MultiLens.Utils.constants as const

class NFW(object):
    """
    this class contains functions concerning the NFW profile

    relation are: R_200 = c * Rs
    """
    def __init__(self):
        self.halo_param = HaloParam()

    def function(self, x, y, rho_s, Rs, pos_x, pos_y):
        """
        returns double integral of NFW profile
        """
        # rho_s [h^-2 M_sun/Mpc physical]
        # Rs [Mpc physical]
        x_ = x - pos_x
        y_ = y - pos_y
        R = np.sqrt(x_**2 + y_**2)
        f_ = self.nfwPot(R, Rs, rho_s)
        return f_

    def derivative(self, x, y, rho_s, Rs, pos_x, pos_y):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        x_ = x - pos_x
        y_ = y - pos_y
        R = np.sqrt(x_**2 + y_**2)
        f_x, f_y = self.alpha(R, Rs, rho_s, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, rho_s, Rs, pos_x, pos_y):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_ = x - pos_x
        y_ = y - pos_y
        R = np.sqrt(x_**2 + y_**2)
        f_x, f_y = self.alpha(R, Rs, rho_s, x_, y_)
        alpha = np.sqrt(f_x**2+f_y**2)
        dalpha_dr = self.dalpha_dr(R, Rs, rho_s, x_, y_)
        f_xx = dalpha_dr*x_*x_/R**2 + alpha*y_**2/R**3
        f_yy = dalpha_dr*y_*y_/R**2 + alpha*x_**2/R**3
        f_xy = dalpha_dr*x_*y_/R**2 - alpha*x_*y_/R**3
        return f_xx, f_yy, f_xy

    def all(self, x, y, rho0, Rs, pos_x, pos_y):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        x_ = x - pos_x
        y_ = y - pos_y
        R = np.sqrt(x_**2 + y_**2)
        f_ = self.nfwPot(R, Rs, rho0)
        f_x, f_y = self.alpha(R, Rs, rho0, x_, y_)
        kappa = self.nfw2D(R, Rs, rho0)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_, f_x, f_y, f_xx, f_yy, f_xy

    def nfw3D(self, R, Rs, rho0):
        """
        three dimenstional NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: rho(R) density
        """
        return rho0/(R/Rs*(1+R/Rs)**2)

    def nfw2D(self, R, Rs, rho0):
        """
        projected two dimenstional NFW profile (kappa*Sigma_crit)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :return: Epsilon(R) projected density at radius R
        """
        x = R/Rs
        Fx = self.F(x)
        return 2*rho0*Rs*Fx

    def nfw2D_smoothed(self, R, Rs, rho0, pixscale):
        """
        projected two dimenstional NFW profile with smoothing around the pixel scale
        this routine is ment to better compare outputs to N-body simulations (not ment ot do lensemodelling with it)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param pixscale: pixel scale (same units as R,Rs)
        :type pixscale: float>0
        :return: Epsilon(R) projected density at radius R
        """
        x = R/Rs
        d = pixscale/(2*Rs)
        a = np.empty_like(x)
        x_ = x[x > d]
        upper = x_+d
        lower = x_-d

        a[x > d] = 4*rho0*Rs**3*(self.g(upper)-self.g(lower))/(2*x_*Rs*pixscale)
        a[x < d] = 4*rho0*Rs**3*self.g(d)/((pixscale/2)**2)
        return a

    def nfwPot(self, R, Rs, rho0):
        """
        lensing potential of NFW profile (*Sigma_crit*D_OL**2)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: Epsilon(R) projected density at radius R
        """
        x=R/Rs
        hx=self.h(x)
        return 2*rho0*Rs**3*hx

    def alpha(self, R, Rs, rho_s, x_, y_):
        """
        deflection angel of NFW profile (*Sigma_crit*D_OL) along the projection to coordinate "axis"

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return: Epsilon(R) projected density at radius R
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.000001)
        else:
            R[R == 0] = 0.000001
        x = R/Rs
        gx = self.g_new(x)
        a = 4*rho_s*Rs**3/R * gx
        C = 4*np.pi*const.G/const.c**2/const.Mpc*const.M_sun
        return C*a*x_/R, C*a*y_/R

    def dalpha_dr(self, R, Rs, rho_s, x_, y_):
        """
        computes the radial derivative of the deflection angle (all in units of Mpc or rad)
        :param R:
        :param Rs:
        :param rho_s:
        :param x_:
        :param y_:
        :return:
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.000001)
        else:
            R[R == 0] = 0.000001
        x = R/Rs
        gx = self.g_new(x)
        dgx = self.dg_new(x)
        a = 4*rho_s/x**2*(-Rs*gx +R*dgx)
        C = 4*np.pi*const.G/const.c**2/const.Mpc*const.M_sun
        return a*C

    def nfwGamma(self, R, Rs, rho0, ax_x, ax_y):
        """
        shear gamma of NFW profile (*Sigma_crit) along the projection to coordinate "axis"

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return: Epsilon(R) projected density at radius R
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.001)
        else:
            R[R==0] = 0.001
        x = R/Rs
        gx = self.g(x)
        Fx = self.F(x)
        a = 2*rho0*Rs*(2*gx/x**2 - Fx)#/x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        C = 4*np.pi*const.G/const.c**2/const.Mpc*const.M_sun
        return C*a*(ax_y**2-ax_x**2)/R**2, C*a*2*(ax_x*ax_y)/R**2

    def F_new(self, X):
        """

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                a = 1-2/np.sqrt(1-X**2)*np.arctanh(np.sqrt((1-X)/(1+X)))
            elif X == 1:
                a = 0
            else:  # X > 1:
                a = 1-2/np.sqrt(X**2-1)*np.arctan(np.sqrt((X-1)/(1+X)))
        else:
            a=np.empty_like(X)
            x = X[X < 1]
            a[X < 1] = 1-2/np.sqrt(1-x**2)*np.arctanh(np.sqrt((1-x)/(1+x)))
            a[X == 1] = 0
            x = X[X > 1]
            a[X > 1] = 1-2/np.sqrt(x**2-1)*np.arctan(np.sqrt((x-1)/(1+x)))
        return a

    def dF_new(self, X):
        """
        derivative of F_new
        :param x:
        :return:
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                a = -2*X*np.arctanh(np.sqrt((1-X)/(X+1)))/(1-X**2)**(3./2) + 1/(X*(X+1)*np.sqrt(1-X**2)*np.sqrt((1-X)/(X+1)))
            elif X == 1:
                a = 2./3
            else:  # X > 1:
                a = -1/(X*(X+1)*np.sqrt(X**2-1)*np.sqrt((X-1)/(X+1))) + 2*X*np.arctan(np.sqrt((X-1)/(X+1)))/(X**2-1)**(3./2)
        else:
            a=np.empty_like(X)
            x = X[X < 1]
            a[X < 1] = 1-2/np.sqrt(1-x**2)*np.arctanh(np.sqrt((1-x)/(1+x)))
            a[X == 1] = 0
            x = X[X > 1]
            a[X > 1] = 1-2/np.sqrt(x**2-1)*np.arctan(np.sqrt((x-1)/(1+x)))
        return a

    def g_new(self, x):
        """

        :param x: R/Rs
        :type x: float >0
        """
        g = np.log(x/2.) + 1 - self.F_new(x)
        return g

    def dg_new(self, x):
        """
        derivative of g_new
        :param x: R/Rs
        :type x: float >0
        """
        return 1/x - self.dF_new(x)

    def F(self, X):
        """
        analytic solution of the projection integral

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1 and X > 0:
                a = 1/(X**2-1)*(1-2/np.sqrt(1-X**2)*np.arctanh(np.sqrt((1-X)/(1+X))))
            elif X == 1:
                a = 1./3
            else:  # X > 1:
                a = 1/(X**2-1)*(1-2/np.sqrt(X**2-1)*np.arctan(np.sqrt((X-1)/(1+X))))
        else:
            a=np.empty_like(X)
            x = X[X<1]
            a[X<1] = 1/(x**2-1)*(1-2/np.sqrt(1-x**2)*np.arctanh(np.sqrt((1-x)/(1+x))))
            a[X==1] = 1./3.
            x = X[X>1]
            a[X>1] = 1/(x**2-1)*(1-2/np.sqrt(x**2-1)*np.arctan(np.sqrt((x-1)/(1+x))))
        return a

    def g(self, X):
        """
        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(0.00001, X)
                a = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)
            elif X == 1:
                a = 1 + np.log(1./2.)
            else:  #(X > 1):
                a = np.log(X/2) + 1/np.sqrt(X**2-1)*np.arccos(1./X)
        else:
            a=np.empty_like(X)
            X[X==0] = 0.00001
            x = X[X<1]

            a[X<1] = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)

            a[X==1] = 1 + np.log(1./2.)

            x = X[X>1]
            a[X>1] = np.log(x/2) + 1/np.sqrt(x**2-1)*np.arccos(1./x)
        return a

    def h(self, X):
        """
        analytic solution of integral for NFW profile to compute the potential

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(0.001, X)
                a = np.log(x/2.)**2 - np.arccosh(1./x)**2
            elif X >= 1:
                a = np.log(X/2.)**2 + np.arccos(1./X)**2
            else:
                a = 0
        else:
            a=np.empty_like(X)
            X[X==0] = 0.001
            x = X[(X<1) & (X>0)]
            a[(X<1) & (X>0)] = np.log(x/2.)**2 - np.arccosh(1./x)**2
            x = X[X >= 1]
            a[X >= 1] = np.log(x/2.)**2 + np.arccos(1./x)**2
        return a

    def alpha2rho0(self, phi_E, Rs):
        """
        convert angle at Rs into rho0
        Attention: Wrong units!!!
        """
        return phi_E/(4*Rs**2*(1+np.log(1./2.)))
