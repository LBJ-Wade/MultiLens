from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

import numpy as np

class LensAssembly(object):
    """
    class to arrange all the strong and weak lenses along a line of sight
    """
    def __init__(self):
        self.redshift_array = []
        self.object_array = []

    def add_lens(self, lensObject):
        """

        :param lensObject: object class of lens_object.py
        :return:
        """
        self.redshift_array.append(lensObject.redshift)
        self.object_array.append(lensObject)
        self._arrange_lenses()

    def remove_lens(self, redshift):
        """
        removes a lens at a given redshift, if existing
        :param redshift: redshift of removing object
        :return:
        """
        for i in xrange(len(self.redshift_array), 0, -1):
            z = self.redshift_array[i]
            if z == redshift:
                del self.redshift_array[i]
                del self.object_array[i]

    def print_info(self):
        print("Number of lenses = ", len(self.redshift_array))
        for lens_object in self.object_array:
            lens_object.print_info()

    def _arrange_lenses(self):
        """
        re-arrange the lens orders to increasing redshifts
        :return:
        """
        self.redshift_array, self.object_array = (list(x) for x in zip(*sorted(zip(self.redshift_array, self.object_array))))

    def clear(self):
        """
        remove all the data of the object class
        :return:
        """
        self.redshift_array = []
        self.object_array = []
        print("LensAssembly class cleared. No lens object specified.")

    def main_deflector(self):
        """
        selects main deflector object
        :return:
        """
        for lensObject in self.object_array:
            if lensObject.main is True:
                return lensObject
        raise ValueError("main deflector not found. Please specify one lens object as such to execute this routine!")

    def get_visible_positions(self):
        """
        return list of pos_x, pos_y of the positions of the lenses in the observer frame
        :return: pos_x, pos_y list
        """
        pos_x_list = np.zeros(len(self.object_array))
        pos_y_list = np.zeros_like(pos_x_list)
        for i in range(len(self.object_array)):
            lensObject = self.object_array[i]
            pos_x_list[i], pos_y_list[i] = lensObject.position()
        return pos_x_list, pos_y_list

    def reset_observer_frame(self):
        """
        undo the positional information of the observer
        :return:
        """
        for lens_object in self.object_array:
            lens_object.reset_position()