#! /usr/bin/env python

# Copyright (C) 2016 ETH Zurich, Institute for Astronomy

# System imports
from __future__ import print_function, division, absolute_import, unicode_literals

__author__ = 'sibirrer'


from MultiLens.Cosmo.cosmo import CosmoProp
import MultiLens.Utils.constants as const


class LensObject(object):
    """
    class to specify the deflection caused by this object
    """

    def __init__(self, redshift, type='point_mass', approximation='weak', main=False, observer_frame=True):
        self.redshift = redshift
        self.type = type
        self.approximation = approximation
        self.kwargs_param = dict([])
        self.main = main
        self.observer_frame = observer_frame
        if type == 'point_mass':
            from MultiLens.Profiles.point_mass import PointMass
            self.func = PointMass()
        elif type == 'NFW':
            from MultiLens.Profiles.nfw import NFW
            self.func = NFW()
        elif type == 'SIS':
            from MultiLens.Profiles.SIS import SIS
            self.func = SIS()
        else:
            raise ValueError("lens type %s not valid." % type)
        self.cosmo = CosmoProp()

    def add_info(self, name, data):
        """
        adds info (i.e. parameters of the lens object
        :return:
        """
        if name == 'kwargs_profile':
            self.kwargs_param = data
            if self.observer_frame and 'pos_x' in data and 'pos_y' in data:
                self.pos_x_observer = data['pos_x']*const.arcsec
                self.pos_y_observer = data['pos_y']*const.arcsec
                self.kwargs_param['pos_x'] = self.cosmo.arcsec2phys(data['pos_x'], z=self.redshift)
                self.kwargs_param['pos_y'] = self.cosmo.arcsec2phys(data['pos_y'], z=self.redshift)
        else:
            print("name %s is not a valid info attribute." % name)

    def potential(self, x, y):
        """
        returns the lensing potential of the object
        :param x: x-coordinate of the light ray
        :param y: y-coordinate of the light ray
        :return: potential
        """
        f_ = self.func.function(x, y, **self.kwargs_param)
        return f_

    def deflection(self, x, y):
        """
        returns the deflection of the object
        :param x: x-coordinate of the light ray
        :param y: y-coordinate of the light ray
        :return: delta_x, delta_y
        """
        f_x0, f_y0 = self.func.derivative(0, 0, **self.kwargs_param)
        f_x, f_y = self.func.derivative(x, y, **self.kwargs_param)
        return f_x-f_x0, f_y-f_y0

    def distortion(self, x, y):
        """
        returns the distortion matrix
        :param x: x-coordinate of the light ray
        :param y: y-coordinate of the light ray
        :return:
        """
        f_xx, f_yy, f_xy = self.func.hessian(x, y, **self.kwargs_param)
        return f_xx, f_yy, f_xy

    def position(self):
        """
        returns x_pos, y_pos
        :return:
        """
        if self.observer_frame and hasattr(self, 'pos_x_observer') and hasattr(self, 'pos_y_observer'):
            return self.pos_x_observer, self.pos_y_observer
        else:
            return 0, 0

    def update_position(self, pos_x, pos_y):
        """
        updates the positional information with the new (unlensed) positions
        :param pos_x:
        :param pos_y:
        :return:
        """
        if self.observer_frame:
            self.kwargs_param['pos_x'] = pos_x
            self.kwargs_param['pos_y'] = pos_y

    def reset_position(self):
        """
        reset position to the one of the observer
        :return:
        """
        self.kwargs_param['pos_x'] = self.cosmo.arcsec2phys(self.pos_x_observer/const.arcsec, z=self.redshift)
        self.kwargs_param['pos_y'] = self.cosmo.arcsec2phys(self.pos_y_observer/const.arcsec, z=self.redshift)

    def print_info(self):
        """
        print all the information about the lens
        :return:
        """
        print('==========')
        if self.main is True:
            print("This is the main deflector.")
        print("redshift = ", self.redshift)
        print("type = ", self.type)
        print("approximation: ", self.approximation)
        print("parameters: ", self.kwargs_param)
