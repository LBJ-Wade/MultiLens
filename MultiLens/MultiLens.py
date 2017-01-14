#! /usr/bin/env python

# Copyright (C) 2016 ETH Zurich, Institute for Astronomy

# System imports
from __future__ import print_function, division, absolute_import, unicode_literals


# External modules
import numpy as np

# MultiLens imports
from MultiLens.analytic_lens import AnalyticLens
from MultiLens.Cosmo.cosmo import CosmoProp

class MultiLens(object):
    """
    this class aims to compute the lensing quantities of multi-plane lenses with full ray-tracing and approximation methods
    """

    def __init__(self):
        self.analyticLens = AnalyticLens()
        self.cosmo = CosmoProp()

    def full_ray_tracing(self, lensAssembly, z_source, x_array, y_array, observer_frame=True):
        """
        full ray-tracing routine (eqn 10,11 in Birrer in prep), implemented with equation 12 in a recursive way
        (!assuming flat cosmology!)
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_source: redshift of the source
        :param x_array: x-coords of the rays
        :param y_array: y-coords of the rays
        :return: deflections delta x_coords, delta y_coords such that x_source = x - delta x_source
        """
        if observer_frame:
            self._full_ray_tracing_observer(lensAssembly)
        object_list = lensAssembly.object_array
        alpha_x_tot = x_array.copy()
        alpha_y_tot = y_array.copy()
        x_k = np.zeros_like(alpha_x_tot)
        y_k = np.zeros_like(alpha_x_tot)
        z_last = 0
        for lensObject in object_list:
            z = lensObject.redshift
            if z < z_source:
                T_k_last = self.cosmo.T_xy(z_last, z)
                x_k += alpha_x_tot*T_k_last
                y_k += alpha_y_tot*T_k_last
                x_k_phys, y_k_phys = x_k/(1+z), y_k/(1+z)
                alpha_x, alpha_y = lensObject.deflection(x_k_phys, y_k_phys)
                alpha_x_tot -= alpha_x
                alpha_y_tot -= alpha_y
                z_last = z
            else:
                pass
        T_k_last = self.cosmo.T_xy(z_last, z_source)
        D_s = self.cosmo.D_xy(0, z_source)
        x_k += alpha_x_tot*T_k_last
        y_k += alpha_y_tot*T_k_last
        x_s_phys, y_s_phys = x_k/(1+z_source), y_k/(1+z_source)
        beta_sx = x_s_phys / D_s
        beta_sy = y_s_phys / D_s
        return beta_sx, beta_sy

    def _full_ray_tracing_observer(self, lensAssembly):
        """
        computes the real positions of the lens objects given the position in the observer frame
        :param lensAssembly:
        :return:
        """
        object_list = lensAssembly.object_array
        alpha_x_tot, alpha_y_tot = lensAssembly.get_visible_positions()
        x_k = np.zeros_like(alpha_x_tot)
        y_k = np.zeros_like(alpha_y_tot)
        z_last = 0
        i = 0
        for lensObject in object_list:
            z = lensObject.redshift
            T_k_last = self.cosmo.T_xy(z_last, z)
            x_k += alpha_x_tot*T_k_last
            y_k += alpha_y_tot*T_k_last
            x_k_phys, y_k_phys = x_k/(1+z), y_k/(1+z)
            lensObject.update_position(x_k_phys[i], y_k_phys[i])  # update position of the i'th lens according to the deflection
            alpha_x, alpha_y = lensObject.deflection(x_k_phys, y_k_phys)
            alpha_x_tot -= alpha_x
            alpha_y_tot -= alpha_y
            z_last = z
            i += 1
        return 0

    def combined_ray_tracing(self, lensAssembly, z_source, x_array, y_array, observer_frame=True):
        """
        ray-tracing routine with Born approximation for the objects specified (eqn 17 in Birrer in prep)
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_source: redshift of the source
        :param x_array: x-coords of the rays
        :param y_array: y-coords of the rays
        :return: deflections delta x_coords, delta y_coords such that x_source = x - delta x_source
        """
        if observer_frame:
            self._combined_ray_tracing_observer(lensAssembly, z_source)
        else:
            lensAssembly.reset_observer_frame()
        object_list = lensAssembly.object_array
        mainLens = lensAssembly.main_deflector()
        z_d = mainLens.redshift
        beta_dx = x_array.copy()
        beta_dy = y_array.copy()
        beta_sx = x_array.copy()
        beta_sy = y_array.copy()
        alpha_x_foreground = np.zeros_like(x_array)
        alpha_y_foreground = np.zeros_like(y_array)
        alpha_dx, alpha_dy = 0, 0
        Ds = self.cosmo.D_xy(0, z_source)
        Dd = self.cosmo.D_xy(0, z_d)
        i = 0
        for lensObject in object_list:
            z = lensObject.redshift
            if z < z_d:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                D_kd = self.cosmo.D_xy(z, z_d)
                alpha_x, alpha_y = lensObject.deflection(D_k*x_array, D_k*y_array)
                alpha_x_foreground += alpha_x
                alpha_y_foreground += alpha_y
                beta_sx -= D_ks/Ds*alpha_x
                beta_sy -= D_ks/Ds*alpha_y
                beta_dx -= D_kd/Dd*alpha_x
                beta_dy -= D_kd/Dd*alpha_y
            elif lensObject.main is True:
                D_ds = self.cosmo.D_xy(z_d, z_source)
                alpha_dx, alpha_dy = lensObject.deflection(Dd*beta_dx, Dd*beta_dy)
                beta_sx -= D_ds/Ds*alpha_dx
                beta_sy -= D_ds/Ds*alpha_dy
            elif z >= z_d:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                D_kd = self.cosmo.D_xy(z_d, z)
                beta_x = beta_dx - D_kd/D_k*(alpha_dx + alpha_x_foreground)  # equation 16 in Birrer in prep
                beta_y = beta_dy - D_kd/D_k*(alpha_dy + alpha_y_foreground)  # equation 16 in Birrer in prep
                alpha_x, alpha_y = lensObject.deflection(D_k*beta_x, D_k*beta_y)
                beta_sx -= D_ks/Ds*alpha_x
                beta_sy -= D_ks/Ds*alpha_y
            i += 1
        return beta_sx, beta_sy

    def _combined_ray_tracing_observer(self, lensAssembly, z_source):
        """
        computes the real position of the lensing objects given observer frame coordinates
        :param lensAssembly:
        :return:
        """
        object_list = lensAssembly.object_array
        mainLens = lensAssembly.main_deflector()
        z_d = mainLens.redshift
        x_array, y_array = lensAssembly.get_visible_positions()
        beta_dx = x_array.copy()
        beta_dy = y_array.copy()
        beta_sx = x_array.copy()
        beta_sy = y_array.copy()
        alpha_x_foreground = 0
        alpha_y_foreground = 0
        alpha_dx, alpha_dy = 0, 0
        Ds = self.cosmo.D_xy(0, z_source)
        Dd = self.cosmo.D_xy(0, z_d)
        i = 0
        for lensObject in object_list:
            z = lensObject.redshift
            if z < z_d:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                D_kd = self.cosmo.D_xy(z, z_d)
                lensObject.update_position(D_k*x_array[i], D_k*y_array[i])
                alpha_x, alpha_y = lensObject.deflection(D_k*x_array, D_k*y_array)
                alpha_x_foreground += alpha_x
                alpha_y_foreground += alpha_y
                beta_sx -= D_ks/Ds*alpha_x
                beta_sy -= D_ks/Ds*alpha_y
                beta_dx -= D_kd/Dd*alpha_x
                beta_dy -= D_kd/Dd*alpha_y
            elif lensObject.main is True:
                D_ds = self.cosmo.D_xy(z_d, z_source)
                lensObject.update_position(Dd*x_array[i], Dd*y_array[i])
                alpha_dx, alpha_dy = lensObject.deflection(Dd*beta_dx, Dd*beta_dy)
                alpha_dx *= D_ds/Ds
                alpha_dy *= D_ds/Ds
                beta_sx -= alpha_dx
                beta_sy -= alpha_dy
            elif z >= z_d:
                D_k = self.cosmo.D_xy(0, z)
                D_kd = self.cosmo.D_xy(z_d, z)
                beta_x = beta_dx - D_kd/D_k*(alpha_dx + alpha_x_foreground)  # equation 16 in Birrer in prep
                beta_y = beta_dy - D_kd/D_k*(alpha_dy + alpha_y_foreground)  # equation 16 in Birrer in prep
                lensObject.update_position(D_k*beta_x[i], D_k*beta_y[i])
            i += 1
        return 0

    def born_ray_tracing(self, lensAssembly, z_source, x_array, y_array):
        """
        routine with Born approximation for all objects (eqn 14 in Birrer in prep)
        :param object_list: list of sources with specified physical deflection angles (sorted by redshift)
        :param z_source: redshift of the source
        :param x_array: x-coords of the rays
        :param y_array: y-coords of the rays
        :return: deflections delta x_coords, delta y_coords such that x_source = x - delta x_source
        """
        lensAssembly.reset_observer_frame()
        object_list = lensAssembly.object_array
        beta_sx = x_array.copy()
        beta_sy = y_array.copy()
        Ds = self.cosmo.D_xy(0, z_source)
        for lensObject in object_list:
            z = lensObject.redshift
            if z < z_source:
                D_k = self.cosmo.D_xy(0, z)
                D_ks = self.cosmo.D_xy(z, z_source)
                delta_x, delta_y = lensObject.deflection(D_k*x_array, D_k*y_array)
                beta_sx -= delta_x*D_ks/Ds
                beta_sy -= delta_y*D_ks/Ds
        return beta_sx, beta_sy

    def analytic_mapping(self, lensAssembly, z_source, x_array, y_array, LOS_corrected=True, observer_frame=True):
        """
        computes equation 29 in Birrer in prep with analytic terms for the LOS structure
        :param object_list:
        :param z_source:
        :param x_array:
        :param y_array:
        :return:
        """
        if observer_frame:
            self._full_ray_tracing_observer(lensAssembly)
        else:
            lensAssembly.reset_observer_frame()
        object_list = lensAssembly.object_array
        mainLens = lensAssembly.main_deflector()
        z_d = mainLens.redshift
        D_ds = self.cosmo.D_xy(z_d, z_source)
        Ds = self.cosmo.D_xy(0, z_source)
        Dd = self.cosmo.D_xy(0, z_d)
        gamma_A = self.analyticLens.shear_lens(object_list, z_d)
        gamma_B = self.analyticLens.shear_foreground(object_list, z_lens=z_d, z_source=z_source)
        if LOS_corrected is True:
            gamma_C = self.analyticLens.shear_background_first_order(object_list, z_d, z_source)
        else:
            gamma_C = self.analyticLens.shear_background_zero(object_list, z_d, z_source)
        gamma_BC = gamma_B + gamma_C
        x_lens = gamma_A[0][0]*x_array + gamma_A[0][1]*y_array + x_array
        y_lens = gamma_A[1][0]*x_array + gamma_A[1][1]*y_array + y_array
        shear_x = gamma_BC[0][0]*x_array + gamma_BC[0][1]*y_array
        shear_y = gamma_BC[1][0]*x_array + gamma_BC[1][1]*y_array

        alpha_x, alpha_y = mainLens.deflection(Dd*x_lens, Dd*y_lens)
        beta_sx = x_array - D_ds/Ds * alpha_x + shear_x
        beta_sy = y_array - D_ds/Ds * alpha_y + shear_y
        return beta_sx, beta_sy

    def analytic_matrices(self, lensAssembly, z_source, LOS_corrected=True, observer_frame=True):
        """
        computes equation 29 in Birrer in prep with analytic terms for the LOS structure
        :param object_list:
        :param z_source:
        :param x_array:
        :param y_array:
        :return:
        """
        if observer_frame:
            self._full_ray_tracing_observer(lensAssembly)
        else:
            lensAssembly.reset_observer_frame()
        object_list = lensAssembly.object_array
        mainLens = lensAssembly.main_deflector()
        z_d = mainLens.redshift
        gamma_A = self.analyticLens.shear_lens(object_list, z_d)
        gamma_B = self.analyticLens.shear_foreground(object_list, z_lens=z_d, z_source=z_source)
        if LOS_corrected is True:
            gamma_C = self.analyticLens.shear_background_first_order(object_list, z_d, z_source)
        else:
            gamma_C = self.analyticLens.shear_background_zero(object_list, z_d, z_source)
        gamma_BC = gamma_B + gamma_C

        return gamma_A, gamma_BC