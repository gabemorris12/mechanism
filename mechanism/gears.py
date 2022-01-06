import csv
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .dataframe import Data
from .vectors import APPEARANCE


class SpurGear:
    with open(APPEARANCE, 'r') as f:
        appearance = json.load(f)

    gear_appearance = appearance['gear_plot']

    def __init__(self, N=0, pd=0, d=0, pressure_angle=20, agma=False, a=0, b=0, backlash=0, ignore_warning=False,
                 size=1000):
        """
        In order to fully define the gear,
        - at least two of the following should be defined: N, pd, and d
        - pressure angle needs to be declared (default value is 20)
        - addendum and dedendum needs to be defined (declaring agma to be true will automatically do this)

        If backlash is set to zero, and agma is set to True, a non-conservative approximation of the backlash will be
        calculated. If it is desired to override the calculated value of the backlash when agma is set to true, then
        provide a value of the backlash that is not equal to zero.

        :param N: The number of teeth of the gear
        :param pd: The diametral pitch of the gear
        :param d: The pitch diameter of the gear
        :param pressure_angle: The angle at which the line of action propagates in the gear mesh
        :param agma: If true, then the addendum, dedendum, and tooth thickness will be calculated according to their
                     standards
        :param a: The addendum of the tooth; the radial distance above the pitch radius
        :param b: The dedendum of the tooth; the radial distance below the pitch radius; the difference between the
                  dedendum and the addendum are the clearance
        :param backlash: This is the difference between the space width and the tooth thickness. Both the space width
                         and the tooth thickness are arc length measurements along the pitch circle. If backlash is not
                         specified and agma is set to true, the backlash will give a non-conservative estimate for the
                         backlash. Backlash, however, can be specified and agma set to true.
        :param size: The size of one involute curve; make this smaller in some cases if SolidWorks says that the points
                     are too close. Default value is 1000.

        Instance Attributes
        -------------------
        r_base: The base radius of the tooth
        ra: The addendum radius of the tooth
        rb: The dedendum radius of the tooth
        tooth_thickness: The arc length of the tooth about the pitch circle
        space_width: The arc length between two teeth about the pitch circle

        The following are np.ndarrays of complex numbers; use np.real() and np.imag() to get x and y components:
        involute_points: The involute curve in the vertical position
        involute_reflection: The involute curve points reflected about the y-axis
        tooth_profile: All the coordinates of the tooth starting on the right side

        Useful Methods to Mention
        -------------------------
        plot: Shows a plot of the tooth profile
        save_coordinates: Saves the coordinates to a file
        rundown: Prints out a table of the key properties of the tooth
        """
        self.pressure_angle = np.deg2rad(pressure_angle)
        assert isinstance(N, int), 'Number of teeth must be an integer.'

        if N and pd:
            self.N, self.pd = N, pd
            self.d = N/pd
        elif N and d:
            self.N, self.d = N, d
            self.pd = N/d
        elif d and pd:
            self.d, self.pd = d, pd
            self.N = pd*d
            if isinstance(self.N, float):
                assert self.N.is_integer(), 'The calculated number of teeth is not registering as an integer. ' \
                                            'Try inputting the number of teeth instead in the case of floating point ' \
                                            'error.'
                self.N = int(self.N)
        else:
            raise Exception('Not enough information given between N, pd, and d.')

        self.r = self.d/2
        self.r_base = self.r*np.cos(self.pressure_angle)

        if agma:
            assert pressure_angle == 20 or pressure_angle == 25, 'AGMA standards only apply to 20 or 25 degree ' \
                                                                 'pressure angles.'
            if self.pd >= 20:
                assert pressure_angle == 20, 'For diametral pitches greater than or equal to 20, the pressure angles ' \
                                             'must be 20 degrees.'
            self.a, self.b = 1/self.pd, 1.25/self.pd
            self.tooth_thickness = 1.571/self.pd
            self.space_width = 2*np.pi*self.r/self.N - self.tooth_thickness
            self.backlash = self.space_width - self.tooth_thickness

            if backlash:
                self.backlash = backlash
                assert backlash >= 0, 'Backlash cannot be negative. If so, there would be interference.'
                self.tooth_thickness = np.pi*self.r/self.N - 1/2*self.backlash
                self.space_width = self.backlash + self.tooth_thickness

            if self.backlash < 0:
                self.backlash = 0.04/self.pd
                self.tooth_thickness = np.pi*self.r/self.N - 1/2*self.backlash
                self.space_width = self.backlash + self.tooth_thickness
        else:
            self.a, self.b = a, b
            self.backlash = backlash
            assert backlash >= 0, 'Backlash cannot be negative. If so, there would be interference.'
            self.tooth_thickness = np.pi*self.r/self.N - 1/2*self.backlash
            self.space_width = self.backlash + self.tooth_thickness

        self.ra, self.rb = self.r + self.a, self.r - self.b

        if self.rb < self.r_base and not ignore_warning:
            warnings.warn('The dedendum circle radius is less than the base circle radius. Undercutting will occur. To '
                          'fix this, make the gear bigger by increasing the pitch diameter or number of teeth. To '
                          'ignore this warning, pass "ignore=True" at the declaration of the gear object.',
                          RuntimeWarning)

        if self.rb >= self.r_base:
            theta_min = np.sqrt((self.rb/self.r_base)**2 - 1)
        else:
            theta_min = 0

        theta_max = np.sqrt((self.ra/self.r_base)**2 - 1)

        # Get the involute curve coming out of the positive x-axis, then rotate the points.
        # Getting the angle for the point at the pitch circle
        theta_pitch = np.sqrt((self.r/self.r_base)**2 - 1)
        x_pitch = self.r_base*np.cos(theta_pitch) + self.r_base*theta_pitch*np.sin(theta_pitch)
        y_pitch = self.r_base*np.sin(theta_pitch) - self.r_base*theta_pitch*np.cos(theta_pitch)
        theta_pitch = np.angle((x_pitch + 1j*y_pitch))

        # Get the angle corresponding to the circular tooth thickness and the amount needed to rotate the tooth
        pitch_angle = self.tooth_thickness/self.r
        rotation = np.pi/2 - theta_pitch - pitch_angle/2

        # Get the involute curve. Create just a linear relationship if the dedendum radius is less than the base radius.
        thetas = np.linspace(theta_min, theta_max, size)
        x = self.r_base*np.cos(thetas) + self.r_base*thetas*np.sin(thetas)
        y = self.r_base*np.sin(thetas) - self.r_base*thetas*np.cos(thetas)
        involute_points = x + 1j*y
        # noinspection PyTypeChecker
        involute_points = np.insert(involute_points, 0, self.rb + 0j) if self.rb < self.r_base else involute_points

        # Rotate the involute curve and get the reflection/addendum circle
        self.involute_points = rotate(involute_points, rotation)
        self.involute_reflection = -1*np.real(self.involute_points) + 1j*np.imag(self.involute_points)
        addendum_start = np.angle(self.involute_points[-1])
        addendum_end = np.angle(self.involute_reflection[-1])
        addendum_circle = self.ra*np.exp(1j*np.linspace(addendum_start, addendum_end, size))

        # Construct the tooth profile
        self.tooth_profile = np.concatenate((self.involute_points, addendum_circle[1:-1],
                                             np.flip(self.involute_reflection)))

    def plot(self, save='', **kwargs):
        """
        Shows a plot of the gear tooth.

        :param save: The filepath to save the plot image to
        :param kwargs: This gets past to figure.savefig()
        """
        between_teeth = 2*np.pi/self.N
        dedendum_angle = np.angle(self.involute_reflection[0]) - np.angle(self.involute_points[0])
        cushion_angle = (between_teeth - dedendum_angle)/2

        angle_start = np.angle(self.involute_points[0]) - cushion_angle
        angle_end = np.angle(self.involute_reflection[0]) + cushion_angle
        dedendum_draw = np.linspace(angle_start, np.angle(self.involute_points[0]), 1000)
        thetas = np.linspace(angle_start, angle_end, 1000)

        base = self.r_base*np.exp(1j*thetas)
        pitch = self.r*np.exp(1j*thetas)
        dedendum = self.rb*np.exp(1j*dedendum_draw)

        fig, ax = plt.subplots()
        ax.plot(np.real(base), np.imag(base), **SpurGear.gear_appearance['base'])
        ax.plot(np.real(pitch), np.imag(pitch), **SpurGear.gear_appearance['pitch'])
        ax.plot(np.real(self.tooth_profile), np.imag(self.tooth_profile), **SpurGear.gear_appearance['tooth'])
        ax.plot(np.real(dedendum), np.imag(dedendum), **SpurGear.gear_appearance['tooth'])
        ax.plot(-1*np.real(dedendum), np.imag(dedendum), **SpurGear.gear_appearance['tooth'])

        ax.set_title('Spur Gear Tooth Profile')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()

        if save:
            fig.savefig(save, **kwargs)

        plt.show()

    def save_coordinates(self, file='', solidworks=False):
        """
        Saves the coordinates to a file.

        :param file: The filepath to save the coordinates to
        :param solidworks: It true, the coordinates will be saved in a format acceptable to SolidWorks
        """
        with open(file, 'w', newline='') as f:
            if solidworks:
                writer = csv.writer(f, delimiter='\t')
                for c_num in self.tooth_profile:
                    writer.writerow([f'{np.real(c_num):.8f}', f'{np.imag(c_num):.8f}', 0])
            else:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['x', 'y'])
                for c_num in self.tooth_profile:
                    writer.writerow([np.real(c_num), np.imag(c_num)])

    def rundown(self):
        """
        Prints a table of the properties of the gear tooth.
        """
        info = [['Number of Teeth (N)', self.N],
                ['Diametral Pitch (pd)', f'{self.pd:.5f}'],
                ['Pitch Diameter (d)', f'{self.d:.5f}'],
                ['Pitch Radius (r)', f'{self.r:.5f}'],
                ['Pressure Angle (phi)', f'{np.rad2deg(self.pressure_angle):.5f}'],
                ['Base Radius', f'{self.r_base:.5f}'],
                ['Addendum (a)', f'{self.a:.5f}'],
                ['Dedendum (b)', f'{self.b:.5f}'],
                ['Circular Tooth Thickness', f'{self.tooth_thickness:.5f}'],
                ['Circular Space Width', f'{self.space_width:.5f}'],
                ['Circular Backlash', f'{self.backlash:.5f}']]

        Data(info, headers=['Property', 'Value']).print(table=True)


class HelicalGear(SpurGear):
    def __init__(self, N=0, pd=0, d=0, helix_angle=0, face_width=0, hand='right', pressure_angle=20, agma=False, a=0,
                 b=0, backlash=0, ignore_warning=False, size=1000):
        SpurGear.__init__(self, N=N, pd=pd, d=d, pressure_angle=pressure_angle, agma=agma, a=a, b=b, backlash=backlash,
                          ignore_warning=ignore_warning, size=size)
        self.helix_angle = np.deg2rad(helix_angle) if hand == 'right' else -np.deg2rad(helix_angle)
        self.face_width = face_width

        theta_start = np.arccos(-self.face_width/(2*self.r*np.tan(np.pi/2 - self.helix_angle)))
        theta_end = np.arccos(self.face_width/(2*self.r*np.tan(np.pi/2 - self.helix_angle)))

        thetas = np.linspace(theta_start, theta_end, size)

        self.x_sw = self.r*np.cos(thetas)
        self.y_sw = self.r*np.sin(thetas)
        self.z_sw = -np.tan(np.pi/2 - self.helix_angle)*self.r*np.cos(thetas)

        self.x_plt = np.copy(self.x_sw)
        self.y_plt = -self.z_sw
        self.z_plt = np.copy(self.y_sw)

        self.tooth_profile_helical_start = rotate(self.tooth_profile, theta_start - np.pi/2)
        self.tooth_profile_helical_end = rotate(self.tooth_profile, theta_end - np.pi/2)

    def plot(self, save='', **kwargs):
        angle_start = np.amin((np.angle(self.tooth_profile_helical_start), np.angle(self.tooth_profile_helical_end)))
        angle_end = np.amax((np.angle(self.tooth_profile_helical_start), np.angle(self.tooth_profile_helical_end)))

        thetas = np.linspace(angle_start, angle_end, 1000)

        pitch = self.r*np.exp(1j*thetas)
        base = self.r_base*np.exp(1j*thetas)

        fig, ax = plt.subplots()
        ax.plot(np.real(self.tooth_profile_helical_start), np.imag(self.tooth_profile_helical_start),
                **HelicalGear.gear_appearance['tooth'])
        ax.plot(np.real(self.tooth_profile_helical_end), np.imag(self.tooth_profile_helical_end), ls='--',
                **HelicalGear.gear_appearance['tooth'])
        ax.plot(np.real(pitch), np.imag(pitch), **HelicalGear.gear_appearance['pitch'])
        ax.plot(np.real(base), np.imag(base), **HelicalGear.gear_appearance['base'])

        ax.set_title('Helical Gear Tooth Profile')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()

        if save:
            fig.savefig(save, **kwargs)

        plt.show()

    def plot3d(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.set_box_aspect([ax.get_box_aspect()[0]]*3)
        ax.plot3D(self.x_plt, self.y_plt, self.z_plt)
        print(ax.get_box_aspect())
        plt.show()

    def save_coordinates(self, file='', solidworks=False):
        name, ext = os.path.splitext(file)
        tooth_profile1 = open(f'{name}-profile1{ext}', 'w', newline='')
        tooth_profile2 = open(f'{name}-profile2{ext}', 'w', newline='')
        tooth_path = open(f'{name}-path{ext}', 'w', newline='')

        if solidworks:
            profile1_writer = csv.writer(tooth_profile1, delimiter='\t')
            profile2_writer = csv.writer(tooth_profile2, delimiter='\t')
            path_writer = csv.writer(tooth_path, delimiter='\t')
            for c_num in self.tooth_profile_helical_start:
                profile1_writer.writerow([f'{np.real(c_num):.8f}', f'{np.imag(c_num):.8f}', f'{self.face_width/2:.8f}'])

            for c_num in self.tooth_profile_helical_end:
                profile2_writer.writerow([f'{np.real(c_num):.8f}', f'{np.imag(c_num):.8f}',
                                          f'{-self.face_width/2:.8f}'])

            for x, y, z in zip(self.x_sw, self.y_sw, self.z_sw):
                path_writer.writerow([f'{x:.8f}', f'{y:.8f}', f'{z:.8f}'])
        else:
            profile1_writer = csv.writer(tooth_profile1, delimiter='\t')
            profile2_writer = csv.writer(tooth_profile2, delimiter='\t')
            path_writer = csv.writer(tooth_path, delimiter='\t')

            profile1_writer.writerow(['x', 'y'])
            profile2_writer.writerow(['x', 'y'])
            path_writer.writerow(['x', 'y', 'z'])

            for c_num in self.tooth_profile_helical_start:
                profile1_writer.writerow([np.real(c_num), np.imag(c_num)])

            for c_num in self.tooth_profile_helical_end:
                profile2_writer.writerow([np.real(c_num), np.imag(c_num)])

            for x, y, z in zip(self.x_sw, self.y_sw, self.z_sw):
                path_writer.writerow([x, y, z])

        tooth_profile1.close()
        tooth_profile2.close()
        tooth_path.close()


def rotate(coords, rotation):
    """
    Rotates a set of complex coordinates.

    :param coords: An np.ndarray of complex numbers
    :param rotation: The angle of rotation of the coordinates in radians
    :return: An np.ndarray of complex numbers that is rotated the amount of 'rotation'
    """
    return np.abs(coords)*np.exp(1j*(np.angle(coords) + rotation))
