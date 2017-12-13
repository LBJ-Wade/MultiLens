from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

from astropy.cosmology import FlatLambdaCDM
import MultiLens.Utils.constants as const


class CosmoProp(object):
    """
    class to compute cosmological distances
    """
    def __init__(self, H0=70, Om0=0.3, Ob0=0.05):
        """

        :param param_file: parameter file for pycosmo
        :return:
        """
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1./(1+z)

    def D_xy(self, z_observer, z_source):
        """
        angular diamter distance in units of Mpc
        :param z_observer: observer
        :param z_source: source
        :return:
        """
        a_S = self.a_z(z_source)
        D_xy = (self.cosmo.comoving_transverse_distance(z_source) - self.cosmo.comoving_transverse_distance(z_observer))*a_S
        return D_xy.value

    def T_xy(self, z_observer, z_source):
        """
        transverse comoving distance in units of Mpc
        :param z_observer: observer
        :param z_source: source
        :return:
        """
        T_xy = self.cosmo.comoving_transverse_distance(z_source) - self.cosmo.comoving_transverse_distance(z_observer)
        return T_xy.value

    def arcsec2phys(self, arcsec, z):
        """
        computes the physical distance in Mpc given angle in arc seconds
        :param arcsec:
        :param z:
        :return:
        """
        return self.D_xy(0, z)*arcsec*const.arcsec