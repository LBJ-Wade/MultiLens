from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

import numpy as np

from MultiLens.Cosmo.cosmo import CosmoProp

class AnalyticLens(object):
    """
    class to compute the analytic terms in Birrer in prep given the lensing objects
    """

    def __init__(self):
        self.cosmo = CosmoProp()

    def shear_lens(self, object_list, z_lens):
        """
        computes \Gamma^{A} matrix, equation 21 in Birrer in prep
        which computes the distortion of the light rays at the main deflector plane
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_lens: redshift of the lens (main deflector)
        :return: 2x2 matrix
        """
        f_xx = 0
        f_xy = 0
        f_yy = 0
        Dd = self.cosmo.D_xy(0, z_lens)
        for lensObject in object_list:
            z = lensObject.redshift
            if z < z_lens:
                D_k = self.cosmo.D_xy(0, z)
                D_kd = self.cosmo.D_xy(z, z_lens)
                f_xx_k, f_yy_k, f_xy_k = lensObject.distortion(0, 0)
                f_xx -= D_k*D_kd/Dd * f_xx_k
                f_yy -= D_k*D_kd/Dd * f_yy_k
                f_xy -= D_k*D_kd/Dd * f_xy_k
        return np.array([[f_xx, f_xy], [f_xy, f_yy]])

    def shear_foreground(self, object_list, z_lens, z_source):
        """
        computes \Gamma^{B} matrix, equation 23 in Birrer in prep,
        which computes the distortion of the light rays between the lens and observer at the source plane
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_lens: redshift of the lens (main deflector)
        :param z_source: redshift of the source
        :return: 2x2 matrix
        """
        f_xx = 0
        f_xy = 0
        f_yy = 0
        Ds = self.cosmo.D_xy(0, z_source)
        for lensObject in object_list:
            z = lensObject.redshift
            if z < z_lens:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                f_xx_k, f_yy_k, f_xy_k = lensObject.distortion(0, 0)
                f_xx -= D_k*D_ks/Ds * f_xx_k
                f_yy -= D_k*D_ks/Ds * f_yy_k
                f_xy -= D_k*D_ks/Ds * f_xy_k
        return np.array([[f_xx, f_xy], [f_xy, f_yy]])

    def shear_background_zero(self, object_list, z_lens, z_source):
        """
        computes \tilde{\Gamma^{C}} matrix, equation 23 in Birrer in prep,
        which computes the distortion of the light rays between the source and the lens at the source plane
        without taking into account the bending of the light rays
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_lens: redshift of the lens (main deflector)
        :param z_source: redshift of the source
        :return: 2x2 matrix
        """
        f_xx = 0
        f_xy = 0
        f_yy = 0
        Ds = self.cosmo.D_xy(0, z_source)
        for lensObject in object_list:
            z = lensObject.redshift
            if z >= z_lens and not lensObject.main:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                f_xx_k, f_yy_k, f_xy_k = lensObject.distortion(0, 0)
                f_xx -= D_k*D_ks/Ds * f_xx_k
                f_yy -= D_k*D_ks/Ds * f_yy_k
                f_xy -= D_k*D_ks/Ds * f_xy_k
        return np.array([[f_xx, f_xy], [f_xy, f_yy]])

    def shear_background_first_order(self, object_list, z_lens, z_source):
        """
        computes \Gamma^{C} matrix, equation 28 in Birrer in prep,
        which computes the distortion of the light rays between the source and the lens at the source plane
        with the approximation that the Einstein ring is thin and the source small
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_lens: redshift of the lens (main deflector)
        :param z_source: redshift of the source
        :return: 2x2 matrix
        """
        f_xx = 0
        f_xy = 0
        f_yy = 0
        Ds = self.cosmo.D_xy(0, z_source)
        D_ds = self.cosmo.D_xy(z_lens, z_source)
        for lensObject in object_list:
            z = lensObject.redshift
            if z >= z_lens and not lensObject.main:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                D_dk = self.cosmo.D_xy(z_lens, z)
                f_xx_k, f_yy_k, f_xy_k = lensObject.distortion(0, 0)
                A = D_k*D_ks/Ds * (1 - D_dk*Ds/(D_k*D_ds))
                f_xx -= A * f_xx_k
                f_yy -= A * f_yy_k
                f_xy -= A * f_xy_k
        return np.array([[f_xx, f_xy], [f_xy, f_yy]])
