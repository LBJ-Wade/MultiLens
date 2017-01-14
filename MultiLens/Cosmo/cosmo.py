from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'


import PyCosmo

import MultiLens.Utils.constants as const

class CosmoProp(object):
    """
    class to compute cosmological distances
    """
    def __init__(self, param_file=None):
        """

        :param param_file: parameter file for pycosmo
        :return:
        """
        if param_file == None:
            param_file = "MultiLens.Cosmo.pycosmo_config_planck2013"
        self.cosmo = PyCosmo.Cosmo(param_file)

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1./(1+z)

    def D_xy(self, z_observer, z_source):
        """
        angular diamter distance
        :param z_observer: observer
        :param z_source: source
        :return:
        """
        a_S = self.a_z(z_source)
        a_O = self.a_z(z_observer)
        return (self.cosmo.background.dist_trans_a(a_S)[0] - self.cosmo.background.dist_trans_a(a_O)[0])*a_S

    def T_xy(self, z_observer, z_source):
        """
        transverse comoving distance
        :param z_observer: observer
        :param z_source: source
        :return:
        """
        a_S = self.a_z(z_source)
        a_O = self.a_z(z_observer)
        return (self.cosmo.background.dist_trans_a(a_S)[0] - self.cosmo.background.dist_trans_a(a_O)[0])

    def arcsec2phys(self, arcsec, z):
        """
        computes the physical distance in Mpc given angle in arc seconds
        :param arcsec:
        :param z:
        :return:
        """
        return self.D_xy(0, z)*arcsec*const.arcsec