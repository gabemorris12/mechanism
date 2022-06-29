import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

from .vectors import APPEARANCE


class Cam:
    def __init__(self, motion=None, degrees=False, omega=0., rotation='ccw', h=0.0062):
        """
        :param motion: Description of motion as a list of tuples. Each tuple must contain 3 items for rising and falling
                       and two items for dwelling. The first item of the tuple is a string equal to "Rise", "Fall", or
                       "Dwell" (not case-sensitive). For rise and fall motion, the second item in the tuple is the
                       distance at which the follower falls or rises. For dwelling, the second item in the tuple is
                       either the time (in seconds) or angle (in degrees) for which the displacement remains constant.
                       The third item in the tuple for rising and falling is equivalent to the second item for dwelling.
        :param degrees: If true, the last item in each tuple of 'motion' will be considered as degree inputs. If false,
                        they will be considered as the time at which the rising, falling, or dwelling occurs. This will
                        be enough to calculate the angular velocity, omega. The angular velocity will have to be given
                        if 'degrees' is set to true.
        :param omega: The angular velocity of the cam. This is always assumed to be constant and should only be given if
                      'degrees' is set to true.
        :param rotation: The direction of the cam rotation. Use 'ccw' for counterclockwise and 'cw' for clockwise. This
                         will affect the animation and profile plots.
        :param h: The step at which the data set gets initialized from 0 to 2*pi radians.

        Instance Attributes
        -------------------
        motion: A list of tuples that describe the desired motion of the cam and follower
        thetas: An np.ndarray from 0 to 2*pi at given step interval
        thetas_d: An np.ndarray of thetas in degrees
        thetas_r: An np.ndarray of thetas going in either the counterclockwise direction (negative values) or clockwise
                  direction (positive values). This is only used to display the profile and retrieve the animation.
        omega: The angular velocity of the cam in radians per second
        rotation: The cam direction as a string ('ccw' or 'cw')
        times: A list of time intervals of the motion. Only exists if 'degrees' is false.
        intervals: A list of angle intervals (in radians) of the motion.
        shifts: A list of the shifts that would occur when implementing the piecewise functions.
        conditions: A list of np.ndarray's of booleans for each interval, corresponding to thetas. All sizes are equal.
        default: A dictionary that gets past to **kwargs when specifying a default appearance in a plot.

        Supported motion types are as follows:
        naive: Naive object (how not to design a cam)
        harmonic: Harmonic object
        cycloidal: Cycloidal object

        More motion types may be created in the future for overall better cam designs.

        Useful Methods to Mention
        -------------------------
        plot: Plots the displacement of a specified motion type
        svaj: Plots the svaj diagram of a specified motion type
        profile: Plots the cam profile of a specified motion type
        get_base_circle: Performs an analysis to appropriately size a cam
        save_coordinates: Saves the coordinates of the cam profile to a file (can be used to create solidworks part)
        get_animation: Retrieves and animation and follower object.
        """
        self.motion = motion
        self.thetas = np.arange(0, 2*np.pi, h)
        self.thetas_d = np.rad2deg(self.thetas)
        self.omega = omega
        self.rotation = rotation

        # I am also assuming that the input will bring the follower back to zero in one full rotation...
        # I am also assuming that the follower height is 0 when theta is 0
        if not degrees:
            self.times = np.array([t[-1] for t in motion], dtype=np.float64)
            self.omega = 2*np.pi/np.sum(self.times)
            self.intervals = self.omega*self.times
        else:
            self.intervals = np.deg2rad([t[-1] for t in motion])
            if omega:
                self.times = self.intervals/omega
            assert np.abs(
                np.sum(self.intervals) - 2*np.pi) < 1e-8, "Given degree increments don't add up to full rotation."

        self.shifts = np.zeros(self.intervals.size)
        self.shifts[0] = self.intervals[0]
        for i in range(1, self.intervals.size):
            self.shifts[i] = self.shifts[i - 1] + self.intervals[i]

        assert all([m[0].lower() in ('dwell', 'rise', 'fall') for m in
                    self.motion]), 'Only "rise", "fall", and "dwell" are acceptable descriptions.'

        self.conditions = self._get_conditions(self.thetas)

        if not self.omega:
            self.naive = Naive(self.motion, self.shifts, self.intervals, self.conditions, self.thetas)
            self.harmonic = Harmonic(self.motion, self.shifts, self.intervals, self.conditions, self.thetas)
            self.cycloidal = Cycloidal(self.motion, self.shifts, self.intervals, self.conditions, self.thetas)
        else:
            self.naive = Naive(self.motion, self.shifts, self.intervals, self.conditions, self.thetas, omega=self.omega)
            self.harmonic = Harmonic(self.motion, self.shifts, self.intervals, self.conditions, self.thetas,
                                     omega=self.omega)
            self.cycloidal = Cycloidal(self.motion, self.shifts, self.intervals, self.conditions, self.thetas,
                                       omega=self.omega)

        with open(APPEARANCE, 'r') as f:
            appearance = json.load(f)

        self.default = appearance['cam_plot']['default']

        assert self.rotation == 'ccw' or self.rotation == 'cw', "Only 'ccw' or 'cw' must be specified for rotation."

        if self.rotation == 'ccw':
            self.thetas_r = self.thetas*-1
        else:
            self.thetas_r = self.thetas

    def _get_conditions(self, t):
        """ Returns a list of conditional arrays"""
        # t is short for theta
        conditions = [np.logical_and(t >= 0, t < self.shifts[0])]
        for i in range(1, self.shifts.size):
            conditions.append(np.logical_and(t >= self.shifts[i - 1], t < self.shifts[i]))
        return conditions

    def plot(self, kind='', grid=True):
        """
        :param kind: The type of motion desired as a string (i.e. 'cycloidal')
        :param grid: If true, the grid will be added to the axes object
        :return: figure and axes objects
        """
        fig, ax = plt.subplots()

        if grid:
            ax.grid(zorder=1)

        if kind == 'all':
            for obj in (self.naive, self.harmonic, self.cycloidal):
                ax.plot(self.thetas_d, obj.S, **obj.appearance)
            ax.legend()
            ax.set_title('All Motion Types')
        else:
            motion_type = self._get_motion_type(kind)
            ax.plot(self.thetas_d, motion_type.S, **self.default)
            ax.set_title(f'{motion_type} Motion')

        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$Displacement$')

        return fig, ax

    def svaj(self, kind=''):
        """
        :param kind: The type of motion desired as a string
        :return: figure and axes objects
        """
        fig, ax = plt.subplots(nrows=4, ncols=1)
        assert self.omega is not None, Exception(
            'You must include omega input in order to know velocity, acceleration, and jerk.')

        motion_type = self._get_motion_type(kind)
        ax[0].set_title(f'{motion_type} Motion')
        ax[0].plot(self.thetas_d, motion_type.S, **self.default)
        ax[1].plot(self.thetas_d, motion_type.V, **self.default)
        ax[2].plot(self.thetas_d, motion_type.A, **self.default)
        ax[3].plot(self.thetas_d, motion_type.J, **self.default)

        ax[0].set_ylabel(r'$Displacement$')
        ax[1].set_ylabel(r'$Velocity$')
        ax[2].set_ylabel(r'$Acceleration$')
        ax[3].set_ylabel(r'$Jerk$')

        fig.set_size_inches(7, 7.75)
        ax[3].set_xlabel(r'$\theta$ (degrees)')

        return fig, ax

    def profile(self, kind='', base=0, show_base=False, roller_radius=0, show_pitch=False, grid=True):
        """
        This will not call ax.legend() because that is not always desired.

        :param kind: The type of motion desired
        :param base: The base circle radius of the cam
        :param show_base: If true, the base circle will be present in the plot
        :param roller_radius: To be used if the pitch curve is desired
        :param show_pitch: If true, the pitch curve will be present in the plot (roller_radius must be given)
        :param grid: If true, the grid will be added to the axes object
        :return: figure and axes object
        """
        fig, ax = plt.subplots()
        if grid:
            ax.grid(zorder=1)

        pitch_line = self.naive.cam_plot['pitch_line']

        if kind == 'all':
            ax.set_title('All Profiles')
            for obj in (self.naive, self.harmonic, self.cycloidal):
                x, y = obj.get_profile(base, self.thetas_r)
                ax.plot(x, y, **obj.appearance)
        else:
            motion_type = self._get_motion_type(kind)
            ax.set_title(f'{motion_type} Motion Profile')
            x, y = motion_type.get_profile(base, self.thetas_r)
            ax.fill(x, y, **motion_type.cam_plot['fill'])
            if show_pitch:
                assert roller_radius, 'Must include the roller radius to show the pitch.'
                x, y = motion_type.get_profile(base + roller_radius, self.thetas_r)
                ax.plot(x, y, **pitch_line)

        if show_base:
            c_nums = base*np.exp(1j*self.thetas)
            ax.plot(np.real(c_nums), np.imag(c_nums), **self.naive.cam_plot['base_circle'])

        ax.set_aspect('equal')

        return fig, ax

    def get_base_circle(self, kind='', follower='', roller_radius=0, eccentricity=0, max_pressure_angle=0,
                        desired_min_rho=0, conservative_flat=False, plot=False):
        """
        :param kind: Type of motion desired for the analysis (i.e. 'harmonic', 'cycloidal').
        :param follower: The type of follower. Either 'roller' or 'flat' are acceptable arguments.
        :param roller_radius: The radius of the roller follower.
        :param eccentricity: The offset of the follower from the center (used for roller followers).
        :param max_pressure_angle: The desired maximum pressure angle (used for roller followers).
        :param desired_min_rho: The desired minimum radius of curvature. Used for flat follower.
        :param conservative_flat: If the follower is a flat follower and this is set to true, then a cam profile with
                                  a positive radius of curvature for the entire profile will be returned. This implies
                                  that the surface of the cam is concave down from 0 to 180 degrees and concave up from
                                  180 to 360 degrees.
        :param plot: Choose whether to include a plot of the pressure angles at the calculated base circle. Only used
                     with roller followers.
        :return: A dictionary of the suggested base circle, minimum radius of curvature at that base circle, and the
                 calculated rho values. Minimum face width is added for a flat-faced follower. The values of phi are
                 also added to the dictionary if "plot" is set to true.
        """
        assert self.omega is not None, Exception(
            'The angular velocity of the cam must be known to conduct this analysis')

        motion_type = self._get_motion_type(kind)
        y = motion_type.S
        y_ = motion_type.V/self.omega
        y__ = motion_type.A/self.omega**2

        if follower == 'roller':
            assert roller_radius is not None and max_pressure_angle is not None, Exception(
                'Roller radius (Rf) and max pressure angle must be included in order to complete analysis.')

            phi = np.deg2rad(max_pressure_angle)
            Rf = roller_radius

            phi_max = lambda Rb: phi - np.max(
                np.abs(np.arctan((y_ - eccentricity)/(y + np.sqrt((Rb + Rf)**2 + eccentricity**2)))))
            base_radius = fsolve(phi_max, np.array([.1]))[0]

            # Checking the minimum radius of curvature
            Rp = base_radius + roller_radius
            rhos = ((Rp + y)**2 + y_**2)**1.5/(
                    (Rp + y)**2 + 2*y_**2 - y__*(Rp + y))
            min_rho = np.min(np.abs(rhos))

            assert min_rho > roller_radius, 'Calculated base radius results in a minimum radius of curvature that is ' \
                                            'less than or equal to the roller radius.'

            if plot:
                fig, ax = plt.subplots()
                phis = np.abs(np.arctan((y_ - eccentricity)/(y + np.sqrt((base_radius + Rf)**2 + eccentricity**2))))
                ax.plot(self.thetas_d, np.rad2deg(phis), label=r'$|\phi(\theta)|$', **self.default)
                ax.plot(self.thetas_d, max_pressure_angle*np.ones(self.thetas.size), color='black', ls='--',
                        label=fr'$\phi={max_pressure_angle}^\circ$')
                ax.set_title(f'Pressure Angles with $R_b={base_radius:.5f}$')
                ax.set_xlabel(r'$\theta$ (deg)')
                ax.set_ylabel(r'$|\phi|$ (deg)')
                ax.grid()
                ax.legend()
                plt.show()
                return {'Rb': base_radius, 'Min Rho': min_rho, 'phis': np.rad2deg(phis), 'rhos': rhos}
            return {'Rb': base_radius, 'Min Rho': min_rho, 'rhos': rhos}
        elif follower == 'flat':
            assert desired_min_rho is not None, 'Minimum rho must be specified.'

            min_face_width = np.max(y_) - np.min(y_)

            if not conservative_flat:
                base_circle = desired_min_rho - np.min(y + y__)
                rhos = base_circle + y + y__

                return {'Rb': base_circle, 'Min Rho': desired_min_rho, 'Min Face Width': min_face_width, 'rhos': rhos}

            Rb_, min_rho, step = 1/2, -1, 0.001
            rhos = None

            while min_rho < desired_min_rho:
                rhos = ((Rb_ + y)**2 + y_**2)**1.5/(
                        (Rb_ + y)**2 + 2*y_**2 - y__*(Rb_ + y))
                min_rho = np.min(rhos)
                Rb_ += step

            return {'Rb': Rb_, 'Min Rho': min_rho, 'Min Face Width': min_face_width, 'rhos': rhos}

        else:
            raise Exception('Only acceptable arguments for follower are "flat" and "roller".')

    def save_coordinates(self, file='', kind='', base=0, solidworks=False):
        """
        :param file: The filepath to save the file too
        :param kind: The desired motion type
        :param base: The base circle radius of the cam
        :param solidworks: If true, the file structure will be acceptable by solidworks standards. Use a .txt file
                           extension for solidworks to be able to select the file.
        """
        motion_type = self._get_motion_type(kind)

        x_coords, y_coords = motion_type.get_profile(base, self.thetas_r)

        with open(file, 'w', newline='') as f:
            if solidworks:
                writer = csv.writer(f, delimiter='\t')
                for x, y in zip(x_coords, y_coords):
                    writer.writerow([f'{x:.8f}', f'{y:.8f}', 0])
            else:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['x', 'y'])
                for x, y in zip(x_coords, y_coords):
                    writer.writerow([x, y])

    def _get_motion_type(self, kind):
        """Returns the motion object specified by 'kind'"""
        if kind == 'naive':
            return self.naive
        elif kind == 'harmonic':
            return self.harmonic
        elif kind == 'cycloidal':
            return self.cycloidal
        else:
            raise Exception('Unknown kind specified.')

    def get_animation(self, kind=None, base=0, inc=10, cushion=0.5, roller_radius=0, face_width=0, length=0, width=0,
                      eccentricity=0, grid=True):
        """
        :param kind: The motion type to base the animation off of
        :param base: The base radius of the cam
        :param inc: Adjusts the speed of the animation by incrementing across the values of thetas.
        :param cushion: The cushion of the window around the objects
        :param roller_radius: The radius of the roller follower. If specified, the animation returns a roller animation.
        :param face_width: The face_width of the follower. If specified, the animation returns a flat faced follower
                           animation
        :param length: The length of the follower (optional)
        :param width: The width of the follower (optional)
        :param eccentricity: The offset of the follower
        :param grid: If true, the grid will be added to the axes object
        :return: animation, figure, axes, and follower object
        """
        motion_type = self._get_motion_type(kind)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_title(f'{motion_type} Animation')
        if grid:
            ax.grid(zorder=1)

        if roller_radius:
            follower = RollerFollower(motion_type, base, self.thetas_r, inc, roller_radius=roller_radius, length=length,
                                      width=width, eccentricity=eccentricity)
            items = [ax.fill([], [], **motion_type.cam_plot['fill'])[0],
                     ax.fill([], [], **motion_type.cam_plot['follower_fill'])[0],
                     ax.fill([], [], **motion_type.cam_plot['follower_fill'])[0]]
        elif face_width:
            follower = FlatFollower(motion_type, base, self.thetas_r, inc, face_width=face_width, length=length,
                                    width=width, eccentricity=eccentricity)
            items = [ax.fill([], [], **motion_type.cam_plot['fill'])[0],
                     ax.fill([], [], **motion_type.cam_plot['follower_fill'])[0]]
        else:
            raise Exception('A roller radius or face width must be specified.')

        (x_min, x_max), (y_min, y_max) = follower.get_bounds()

        ax.set_xlim(x_min - cushion, x_max + cushion)
        ax.set_ylim(y_min - cushion, y_max + cushion)

        def init():
            return items

        def animate_roller(index):
            items[0].set_xy(np.stack((follower.cam_x[index], follower.cam_y[index]), axis=1))
            items[1].set_xy(np.stack((follower.roller_x[index], follower.roller_y[index]), axis=1))
            items[2].set_xy(follower.vertices[index])
            return items

        def animate_flat(index):
            items[0].set_xy(np.stack((follower.cam_x[index], follower.cam_y[index]), axis=1))
            items[1].set_xy(follower.vertices[index])
            return items

        if roller_radius:
            animate = animate_roller
        else:
            animate = animate_flat

        # noinspection PyTypeChecker
        return FuncAnimation(fig, animate, frames=range(follower.motion_length), interval=20, blit=True,
                             init_func=init), fig, ax, follower


class Motion:
    def __init__(self, motion, shifts, intervals, conditions, thetas, omega=None):
        """
        :param motion: The same list of tuples described in the Cam class documentation
        :param shifts: The shifts from class Cam
        :param intervals: The intervals from class Cam
        :param conditions: The conditions from class Cam
        :param thetas: The np.ndarray from class Cam
        :param omega: The angular velocity of the cam

        Instance Attributes
        -------------------
        motion, shifts, intervals, thetas, and omega are all the same as described in Cam documentation
        S: The displacements of the cam corresponding to each value of theta
        V: The velocity of the cam corresponding to each value of theta
        A: The acceleration of the cam corresponding to each value of theta
        J: The jerk of the cam corresponding to each value of theta

        Methods used are primarily used for internal use to aid the useful methods within the Cam class.
        """
        self.motion = motion
        self.shifts, self.intervals = shifts, intervals
        self.conditions = conditions
        self.thetas = thetas
        self.omega = omega

        self.S = np.piecewise(self.thetas, condlist=self.conditions, funclist=self._get_functions(self.f))
        self.V, self.A, self.J = None, None, None
        if self.omega is not None:
            self._get_rates()

        with open(APPEARANCE, 'r') as f:
            appearance = json.load(f)

        self.cam_plot = appearance['cam_plot']

    def _get_functions(self, func_maker, rate=False):
        """
        :param func_maker: A function that returns a lambda expression
        :param rate: If true, then dwells will be zero
        :return: A list of lambda expressions corresponding to each interval of rotation. Used to create the piecewise
                 function.
        """
        start = self.motion[0]
        if start[0].lower() == 'dwell':
            h1, h2 = 0, 0
            functions = [_dwell_maker(h1)]
        else:
            h1, h2 = 0, start[1]
            functions = [func_maker(h1, h2, self.intervals[0], 0)]
            h1 += h2
            assert start[0].lower() != 'fall', "Displacement must be positive for all rotation; therefore, it can't " \
                                               "with a fall."

        for i in range(1, self.shifts.size):
            motion = self.motion[i]
            if motion[0].lower() == 'dwell' and not rate:
                functions.append(_dwell_maker(h1))
            elif motion[0].lower() == 'dwell' and rate:
                functions.append(_dwell_maker(0))
            else:
                h2 = motion[1] if motion[0].lower() == 'rise' else -motion[1]
                functions.append(func_maker(h1, h2, self.intervals[i], self.shifts[i - 1]))
                h1 += h2
        return functions

    def _get_rates(self):
        """Gets the rates of velocity, acceleration, and jerk of the cam"""
        self.V = np.piecewise(self.thetas, condlist=self.conditions,
                              funclist=self._get_functions(self.f_, rate=True))*self.omega
        self.A = np.piecewise(self.thetas, condlist=self.conditions,
                              funclist=self._get_functions(self.f__, rate=True))*self.omega**2
        self.J = np.piecewise(self.thetas, condlist=self.conditions,
                              funclist=self._get_functions(self.f___, rate=True))*self.omega**3

    def get_profile(self, base, thetas):
        """
        :param base: The base radius of the cam
        :param thetas: This is thetas_r for the Cam class
        :return: The coordinates of the cam
        """
        c_nums = (base + self.S)*np.exp(1j*thetas)
        return np.real(c_nums), np.imag(c_nums)

    @staticmethod
    def f(*args):
        pass

    @staticmethod
    def f_(*args):
        pass

    @staticmethod
    def f__(*args):
        pass

    @staticmethod
    def f___(*args):
        pass


class Naive(Motion):
    def __init__(self, motion, shifts, intervals, conditions, thetas, omega=None):
        Motion.__init__(self, motion, shifts, intervals, conditions, thetas, omega=omega)
        self.appearance = self.cam_plot['naive_line']

    @staticmethod
    def f(h1, h2, B, s):
        return lambda theta: h2/B*(theta - s) + h1

    @staticmethod
    def f_(_, h2, B, __):
        return lambda theta: h2/B

    @staticmethod
    def f__(*_):
        return lambda theta: 0

    @staticmethod
    def f___(*_):
        return lambda theta: 0

    def __str__(self):
        return 'Naive'


class Harmonic(Motion):
    def __init__(self, motion, shifts, intervals, conditions, thetas, omega=None):
        Motion.__init__(self, motion, shifts, intervals, conditions, thetas, omega=omega)
        self.appearance = self.cam_plot['harmonic_line']

    @staticmethod
    def f(h1, h2, B, s):
        return lambda theta: h1 + h2*(1 - np.cos(np.pi*(theta - s)/B))/2

    @staticmethod
    def f_(_, h2, B, s):
        return lambda theta: np.pi*h2*np.sin(np.pi*(theta - s)/B)/(2*B)

    @staticmethod
    def f__(_, h2, B, s):
        return lambda theta: np.pi**2*h2*np.cos(np.pi*(theta - s)/B)/(2*B**2)

    @staticmethod
    def f___(_, h2, B, s):
        return lambda theta: -np.pi**3*h2*np.sin(np.pi*(theta - s)/B)/(2*B**3)

    def __str__(self):
        return 'Harmonic'


class Cycloidal(Motion):
    def __init__(self, motion, shifts, intervals, conditions, thetas, omega=None):
        Motion.__init__(self, motion, shifts, intervals, conditions, thetas, omega=omega)
        self.appearance = self.cam_plot['cycloidal_line']

    @staticmethod
    def f(h1, h2, B, s):
        return lambda theta: h1 + h2*((theta - s)/B - 1/(2*np.pi)*np.sin(2*np.pi*(theta - s)/B))

    @staticmethod
    def f_(_, h2, B, s):
        return lambda theta: h2*(-np.cos(2*np.pi*(theta - s)/B)/B + 1/B)

    @staticmethod
    def f__(_, h2, B, s):
        return lambda theta: 2*np.pi*h2*np.sin(2*np.pi*(theta - s)/B)/B**2

    @staticmethod
    def f___(_, h2, B, s):
        return lambda theta: 4*np.pi**2*h2*np.cos(2*np.pi*(theta - s)/B)/B**3

    def __str__(self):
        return 'Cycloidal'


class RollerFollower:
    def __init__(self, motion, base, thetas, inc, roller_radius, length=0, width=0, eccentricity=0):
        """
        :param motion: Motion object
        :param base: Base radius of the cam
        :param thetas: thetas_r associated with the cam
        :param inc: The increment across the data set to be used
        :param roller_radius: The radius of the follower
        :param length: The length of the follower
        :param width: The width of the follower
        :param eccentricity: The offset of the follower

        Useful Instance Attributes
        --------------------------
        S: The displacement of the follower (np.gradient to get the velocity and acceleration)

        Useful Methods to Mention
        -------------------------
        plot: Plots the displacement alongside the cam displacement to show the difference between the two
        """
        self.indexes = range(0, thetas.size, inc)
        self.motion_length = len(self.indexes)
        self.motion = motion

        h = thetas[0] - thetas[inc]
        start_point = (base, eccentricity)
        width = roller_radius if not width else width
        length = 1.25*base if not length else length

        circle = roller_radius*np.exp(1j*motion.thetas)
        circle_x, circle_y = np.real(circle), np.imag(circle)

        self.cam_x, self.cam_y = [], []  # A list of x and y coordinates of the cam profile for each rotation
        self.roller_centers = []  # A list of x coordinates of the roller center
        self.roller_x, self.roller_y = [], []  # A list of x and y coordinates for the circle of the roller
        self.vertices = []  # Vertices of the follower

        for i in range(self.motion_length):
            cam_profile = motion.get_profile(base, thetas + i*h)
            self.cam_x.append(cam_profile[0])
            self.cam_y.append(cam_profile[1])
            roller_center = _move_circle(cam_profile, roller_radius, start_point)
            self.roller_centers.append(roller_center)
            self.roller_x.append(circle_x + roller_center)
            self.roller_y.append(circle_y + eccentricity)
            start_point = (roller_center, eccentricity)
            self.vertices.append([
                np.array([roller_center, eccentricity + width/2]),
                np.array([roller_center, eccentricity - width/2]),
                np.array([roller_center + length - 0.25, eccentricity - width/2]),
                np.array([roller_center + length, eccentricity]),
                np.array([roller_center + length - 0.25, eccentricity + width/2])
            ])

        self.S = np.array(self.roller_centers) - np.min(self.roller_centers)

    def get_bounds(self):
        """
        :return: The bounds of the animation
        """
        x_max = np.amax(self.vertices)
        x_min = np.amin(self.cam_x)
        y_min, y_max = np.amin(self.cam_y), np.amax(self.cam_y)
        return (x_min, x_max), (y_min, y_max)

    def plot(self, grid=True):
        """
        Plots the displacement of the follower alongside the cam displacement

        :param grid: If true, the grid will be added to the axes object
        :return: figure and axes object"""
        fig, ax = plt.subplots()
        if grid:
            ax.grid(zorder=1)
        ax.plot(np.rad2deg(self.motion.thetas[self.indexes]), self.S, label='Follower Displacement',
                **self.motion.cam_plot['default'])
        ax.plot(np.rad2deg(self.motion.thetas), self.motion.S, **self.motion.appearance)
        ax.set_title('Follower Displacement')
        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$Displacement$')
        ax.legend()
        return fig, ax


class FlatFollower:
    """
    See the documentation for RollerFollower
    """

    def __init__(self, motion, base, thetas, inc, face_width=0, length=0, width=0, eccentricity=0):
        self.indexes = range(0, thetas.size, inc)
        self.motion_length = len(self.indexes)
        self.motion = motion

        h = thetas[0] - thetas[inc]
        width = 1/8*face_width if not width else width
        length = 2*face_width if not length else length

        self.cam_x, self.cam_y = [], []
        self.x_positions = []
        self.vertices = []

        for i in range(self.motion_length):
            x, y = motion.get_profile(base, thetas + i*h)
            self.cam_x.append(x)
            self.cam_y.append(y)
            possible = np.logical_and(y >= -(face_width/2 - eccentricity), y <= face_width/2 + eccentricity)
            x_position = np.max(x[possible])
            self.x_positions.append(x_position)
            self.vertices.append([
                np.array([x_position, face_width/2 + eccentricity]),
                np.array([x_position, -(face_width/2 - eccentricity)]),
                np.array([x_position + width, -(face_width/2 - eccentricity)]),
                np.array([x_position + width, -(face_width/2 - eccentricity) + face_width/2 - width/2]),
                np.array([x_position + length, -(face_width/2 - eccentricity) + face_width/2 - width/2]),
                np.array([x_position + length, -(face_width/2 - eccentricity) + face_width/2 + width/2]),
                np.array([x_position + width, -(face_width/2 - eccentricity) + face_width/2 + width/2]),
                np.array([x_position + width, face_width/2 + eccentricity])
            ])

        self.S = np.array(self.x_positions) - np.min(self.x_positions)

    def get_bounds(self):
        x_max = np.amax(self.vertices)
        x_min = np.amin(self.cam_x)
        y_min, y_max = np.amin(self.cam_y), np.amax(self.cam_y)
        return (x_min, x_max), (y_min, y_max)

    def plot(self, grid=True):
        fig, ax = plt.subplots()
        if grid:
            ax.grid(zorder=1)
        ax.plot(np.rad2deg(self.motion.thetas[self.indexes]), self.S, label='Follower Displacement',
                **self.motion.cam_plot['default'])
        ax.plot(np.rad2deg(self.motion.thetas), self.motion.S, **self.motion.appearance)
        ax.set_title('Follower Displacement')
        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$Displacement$')
        ax.legend()

        return fig, ax


def _dwell_maker(h):
    return lambda theta: h


def _move_circle(cam_profile, r, start_point):
    """
    This function is responsible for keeping the roller tangent to the surface of the cam within a 1/1000th unit
    tolerance.

    :param cam_profile: The coordinates of the cam in a (2, N) format.
    :param r: The roller radius
    :param start_point: The point to start the circle at
    :return: The center of the circle in the x direction to where the circle is tangent to the surface of the cam
    """
    cam_x, cam_y = cam_profile
    a, b = start_point
    inside = np.logical_and(cam_y < np.sqrt(r**2 - (cam_x - a)**2) + b, cam_y > -np.sqrt(r**2 - (cam_x - a)**2) + b)

    while True:
        if np.any(inside):
            a += 0.001
            np.logical_and(cam_y < np.sqrt(r**2 - (cam_x - a)**2) + b, cam_y > -np.sqrt(r**2 - (cam_x - a)**2) + b,
                           out=inside)
        else:
            a -= 0.001
            np.logical_and(cam_y < np.sqrt(r**2 - (cam_x - a)**2) + b, cam_y > -np.sqrt(r**2 - (cam_x - a)**2) + b,
                           out=inside)
            if np.any(inside):
                a += 0.001
                return a
