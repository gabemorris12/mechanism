import csv
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

from .dataframe import Data
from .vectors import APPEARANCE


class SpurGear:
    with open(APPEARANCE, 'r') as f:
        appearance = json.load(f)

    gear_appearance = appearance['gear_plot']

    def __init__(self, N=0, pd=0, d=0, pressure_angle=20, agma=False, a=0, b=0, backlash=0, internal=False,
                 ignore_undercut=False, size=300):
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
        :param internal: If true, the involute will be based off an internal gear, in which the teeth are holes and
                         everything around it gets filled. Internal gears are the inverse of external gears. When this
                         happens, backlash needs to make the tooth profile wider instead of smaller. Also, the dedendum
                         and addendum circles are switched. See Shigley's Machine Design 11th Edition page 688.
        :param ignore_undercut: If true, the undercut warning will be ignored. This occurs when the dedendum is large
                                enough to extend passed the base circle.
        :param size: The size of one involute curve; make this smaller in some cases if SolidWorks says that the points
                     are too close. Default value is 300.

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

        self.internal = internal

        if agma:
            assert pressure_angle == 20 or pressure_angle == 25, 'AGMA standards only apply to 20 or 25 degree ' \
                                                                 'pressure angles.'
            if self.pd >= 20:
                assert pressure_angle == 20, 'For diametral pitches greater than or equal to 20, the pressure angles ' \
                                             'must be 20 degrees.'
            self.a, self.b = 1/self.pd, 1.25/self.pd
            # Using this from the text sometimes results in negative backlashes, which doesn't make sense with external
            # gears. This comes from Design of Machinery, but I'm not convinced that this should be used. More research
            # is required.
            # self.tooth_thickness = 1.571/self.pd
            # self.space_width = 2*np.pi*self.r/self.N - self.tooth_thickness
            # self.backlash = self.space_width - self.tooth_thickness

            if not internal:
                self.backlash = 0.04/self.pd
                self.tooth_thickness = np.pi*self.r/self.N - 1/2*self.backlash
                self.space_width = self.backlash + self.tooth_thickness
            else:
                self.backlash = 0.04/self.pd
                self.tooth_thickness = np.pi*self.r/self.N + 1/2*self.backlash
                self.space_width = -self.backlash + self.tooth_thickness

            if backlash:
                self.backlash = backlash
                # A negative backlash would imply that the user wants an internal/ring gear. If this is the case, then
                # widening the profile is desired, since the space for a normal gear is now void.
                assert backlash >= 0, 'Use a positive value for backlash.'
                backlash = backlash if not internal else -backlash
                self.tooth_thickness = np.pi*self.r/self.N - 1/2*backlash
                self.space_width = backlash + self.tooth_thickness
        else:
            self.a, self.b = a, b
            assert a > 0 and b > 0, 'Addendum and dedendum need to be defined if the agma argument is not present.'
            self.backlash = backlash
            assert backlash >= 0, 'Use a positive value for backlash.'
            backlash = backlash if not internal else -backlash
            self.tooth_thickness = np.pi*self.r/self.N - 1/2*backlash
            self.space_width = backlash + self.tooth_thickness

        if not internal:
            ra, rb = self.r + self.a, self.r - self.b
            self.ra, self.rb = self.r + self.a, self.r - self.b
        else:
            # rb and ra are not the actual values of the addendum and dedendum circles, but are presented here to make
            # the math work out without adding a considerable amount of conditional statements.
            ra, rb = self.r + self.b, self.r - self.a
            self.ra, self.rb = self.r - self.a, self.r + self.b

        if rb < self.r_base and not ignore_undercut:
            warnings.warn('The dedendum circle radius is less than the base circle radius. Undercutting will occur. To '
                          'fix this, make the gear bigger by increasing the pitch diameter or number of teeth. To '
                          'ignore this warning, pass "ignore_undercut=True" at the declaration of the gear object.',
                          RuntimeWarning)

        # Get the involute curve coming out of the positive x-axis, then rotate the points.
        # Getting the angle for the point at the pitch circle
        theta_pitch = np.sqrt((self.r/self.r_base)**2 - 1)
        x_pitch = self._x_inv(theta_pitch)
        y_pitch = self._y_inv(theta_pitch)
        theta_pitch = np.angle((x_pitch + 1j*y_pitch))

        # Get the angle corresponding to the circular tooth thickness and the amount needed to rotate the tooth
        pitch_angle = self.tooth_thickness/self.r
        rotation = np.pi/2 - theta_pitch - pitch_angle/2

        # Get the involute curve. Create just a linear relationship if the dedendum radius is less than the base radius.
        # See the planetary gear branch in the gear bearing repository as this is quite complex to pick up again.
        if rb >= self.r_base:
            theta_min = np.sqrt((rb/self.r_base)**2 - 1)  # Relationship is a result of converting to polar form
            theta_max = np.sqrt((ra/self.r_base)**2 - 1)
            thetas = np.linspace(theta_min, theta_max, size)
            x = self._x_inv(thetas)
            y = self._y_inv(thetas)
        else:
            theta_max = np.sqrt((ra/self.r_base)**2 - 1)
            # Using the absolute value because this does some funky stuff sometimes, but only when there is a greater
            # portion of the gear under the base circle.
            theta_min = np.abs(fsolve(self._solve_theta_min, np.array([theta_max, ]), args=(rb, )))[0]
            theta1 = np.flip(np.linspace(0, theta_min, int(size/2)))
            x1 = 2*self.r_base - self._x_inv(theta1)
            y1 = self._y_inv(theta1)
            theta2 = np.linspace(theta1[0] - theta1[1], theta_max, int(size/2))
            x2 = self._x_inv(theta2)
            y2 = self._y_inv(theta2)
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))

        involute_points = x + 1j*y

        # Rotate the involute curve and get the reflection/addendum circle
        self.involute_points = _rotate(involute_points, rotation)
        self.involute_reflection = -1*np.real(self.involute_points) + 1j*np.imag(self.involute_points)
        addendum_start = np.angle(self.involute_points[-1])
        addendum_end = np.angle(self.involute_reflection[-1])
        addendum_circle = ra*np.exp(1j*np.linspace(addendum_start, addendum_end, size))

        # Construct the tooth profile
        self.tooth_profile = np.concatenate((self.involute_points, addendum_circle[1:-1],
                                             np.flip(self.involute_reflection)))

    def plot(self, grid=True):
        """
        Shows a plot of the gear tooth.
        :param grid: If true, the grid will be added to the axes object
        :return: figure and axes objects
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
        rb = self.r - self.b if not self.internal else self.r - self.a  # Not technically true
        dedendum = rb*np.exp(1j*dedendum_draw)

        fig, ax = plt.subplots()
        if grid:
            ax.grid(zorder=1)
        ax.plot(np.real(base), np.imag(base), **SpurGear.gear_appearance['base'])
        ax.plot(np.real(pitch), np.imag(pitch), **SpurGear.gear_appearance['pitch'])
        ax.plot(np.real(self.tooth_profile), np.imag(self.tooth_profile), **SpurGear.gear_appearance['tooth'])
        ax.plot(np.real(dedendum), np.imag(dedendum), **SpurGear.gear_appearance['tooth'])
        ax.plot(-1*np.real(dedendum), np.imag(dedendum), **SpurGear.gear_appearance['tooth'])

        ax.set_title('Spur Gear Tooth Profile')
        ax.set_aspect('equal')
        ax.legend()

        return fig, ax

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

    def _x_inv(self, thetas_):
        """
        Calculates the x values of an involute curve.
        """
        return self.r_base*np.cos(thetas_) + self.r_base*thetas_*np.sin(thetas_)

    def _y_inv(self, thetas_):
        """
        Calculates the y values of an involute curve.
        """
        return self.r_base*np.sin(thetas_) - self.r_base*thetas_*np.cos(thetas_)

    def _solve_theta_min(self, theta_, rb_):
        """
        Solves for the value of theta min when the base circle is larger than the dedendum circle.
        """
        x_prime_ = 2*self.r_base - self._x_inv(theta_)
        y_ = self._y_inv(theta_)
        return np.abs(x_prime_ + 1j*y_) - rb_


def _rotate(coords, rotation):
    """
    Rotates a set of complex coordinates.

    :param coords: An np.ndarray of complex numbers
    :param rotation: The angle of rotation of the coordinates in radians
    :return: An np.ndarray of complex numbers that is rotated the amount of 'rotation'
    """
    return np.abs(coords)*np.exp(1j*(np.angle(coords) + rotation))
