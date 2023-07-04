import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy.optimize import fsolve

from .dataframe import Data
from .vectors import VectorBase, APPEARANCE
from .player import Player


# going = True


class Joint:
    follow_all = False

    def __init__(self, name='', follow=None, style=None, exclude=None, vel_arrow_kwargs=None, acc_arrow_kwargs=None,
                 **kwargs):
        """
        :param: name: str; The name of the joint. Typically, a capital letter.
        :param: follow: bool; If true, the path of the joint will be drawn in the animation.
        :param: style: str; A named style located in the appearance json and under the 'joint_path' option.
        :param: exclude: bool; If true, the velocity and acceleration arrows will not be displayed in plots.
        :param: vel_arrow_kwargs: dict; kwargs to be passed into the FancyArrowPatch that makes up the velocity arrows.
        :param: acc_arrow_kwargs: dict; kwargs to be passed into the FancyArrowPatch that makes up the acceleration
                arrows.
        :param: kwargs: Extra arguments that get past to plt.plot(). Useful only if follow is set to true.

        Instance Variables
        ------------------
        x_pos, y_pos: The global x and y position of the joint
        x_vel, y_vel: The global x and y velocity components of the joint.
        x_acc, y_acc: The global x and y acceleration components of the joint.

        x_positions, y_positions: An ndarray consisting of the x and y positions. Values get populated only when
            the iterate() method gets called.
        x_velocities, y_velocities: An ndarray consisting of the x and y velocities. Values get populated only when
            the iterate() method gets called.
        x_accelerations, y_accelerations: An ndarray consisting of the x and y accelerations. Values get populated only
            when the iterate() method gets called.
        vel_mags, vel_angles: An ndarray consisting of the velocity magnitudes and angles. Values get populated only
            when the iterate() method gets called.
        acc_mags, acc_angles: An ndarray consisting of the acceleration magnitudes and angles. Values get populated only
            when the iterate() method gets called.
        """
        self.name = name
        self.x_pos, self.y_pos = None, None
        self.x_vel, self.y_vel = None, None
        self.x_acc, self.y_acc = None, None

        self.x_positions, self.y_positions = None, None
        self.x_velocities, self.y_velocities = None, None
        self.x_accelerations, self.y_accelerations = None, None

        self.vel_mags, self.vel_angles = None, None
        self.acc_mags, self.acc_angles = None, None

        self._x_vel_scaled, self._y_vel_scaled = None, None
        self._x_acc_scaled, self._y_acc_scaled = None, None
        self._x_vel_scales, self._y_vel_scales = None, None
        self._x_acc_scales, self._y_acc_scales = None, None
        self._vel_heads, self._acc_heads = None, None  # array of complex numbers

        if follow is None:
            self.follow = self.follow_all
        else:
            self.follow = follow

        with open(APPEARANCE, 'r') as f:
            appearance = json.load(f)

        if style:
            self.kwargs = appearance['joint_path'][style]
        elif kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = appearance['joint_path']['default']

        if vel_arrow_kwargs:
            self.vel_arrow_kwargs = vel_arrow_kwargs
        elif exclude:
            self.vel_arrow_kwargs = dict(lw=0, mutation_scale=0)
        else:
            self.vel_arrow_kwargs = appearance['vel_arrow']

        if acc_arrow_kwargs:
            self.acc_arrow_kwargs = acc_arrow_kwargs
        elif exclude:
            self.acc_arrow_kwargs = dict(lw=0, mutation_scale=0)
        else:
            self.acc_arrow_kwargs = appearance['acc_arrow']

    def _position_is_fixed(self):
        """
        :return: True if the position is globally defined.
        """
        return False if self.x_pos is None or self.y_pos is None else True

    def _velocity_is_fixed(self):
        """
        :return: True if the velocity is globally defined.
        """
        return False if self.x_vel is None or self.y_vel is None else True

    def _acceleration_is_fixed(self):
        """
        :return: True if the acceleration is globally defined.
        """
        return False if self.x_acc is None or self.y_acc is None else True

    def _fix_position(self, x_pos, y_pos):
        """
        Sets self.x_pos and self.y_pos
        """
        self.x_pos, self.y_pos = x_pos, y_pos

    def _fix_velocity(self, x_vel, y_vel):
        """
        Sets self.x_vel and self.y_vel
        """
        self.x_vel, self.y_vel = x_vel, y_vel
        if abs(self.x_vel) < 1e-10:
            self.x_vel = 0
        if abs(self.y_vel) < 1e-10:
            self.y_vel = 0

    def _fix_acceleration(self, x_acc, y_acc):
        """
        Sets self.x_acc and self.y_acc
        """
        self.x_acc, self.y_acc = x_acc, y_acc
        if abs(self.x_acc) < 1e-10:
            self.x_acc = 0
        if abs(self.y_acc) < 1e-10:
            self.y_acc = 0

    def _clear(self):
        """
        Clears the non-iterable instance variables. This must be called between two different calls of calculate() from
        the mechanism instance.
        """
        self.x_pos, self.y_pos = None, None
        self.x_vel, self.y_vel = None, None
        self.x_acc, self.y_acc = None, None

    def _vel_mag(self):
        """
        :return: A tuple of the magnitude and angle of the velocity of the joint.
        """
        return VectorBase(x=self.x_vel, y=self.y_vel)._get_mag()

    def _acc_mag(self):
        """
        :return: A tuple of the magnitude and angle of the acceleration of the joint.
        """
        return VectorBase(x=self.x_acc, y=self.y_acc)._get_mag()

    def _zero(self, s):
        """
        Zeros the iterable instances of the joint object.

        :param s: The size of the iterable instances
        """
        self.x_positions, self.y_positions = np.zeros(s), np.zeros(s)
        self.x_velocities, self.y_velocities = np.zeros(s), np.zeros(s)
        self.x_accelerations, self.y_accelerations = np.zeros(s), np.zeros(s)

        self.vel_mags, self.vel_angles = np.zeros(s), np.zeros(s)
        self.acc_mags, self.acc_angles = np.zeros(s), np.zeros(s)

    def _set_position_data(self, i):
        """
        Sets the position data at the index i.

        :param i: Index
        """
        self.x_positions[i] = self.x_pos
        self.y_positions[i] = self.y_pos

    def _set_velocity_data(self, i):
        """
        Sets the velocity, vel_mag, and vel_angle data at the index i.

        :param i: Index
        """

        self.x_velocities[i] = self.x_vel
        self.y_velocities[i] = self.y_vel

        mag, angle = self._vel_mag()
        self.vel_mags[i] = mag
        self.vel_angles[i] = angle

    def _set_acceleration_data(self, i):
        """
        Sets the acceleration, acc_mag, and acc_angle data at the index i.

        :param i: Index
        """

        self.x_accelerations[i] = self.x_acc
        self.y_accelerations[i] = self.y_acc

        mag, angle = self._acc_mag()
        self.acc_mags[i] = mag
        self.acc_angles[i] = angle

    def _scale_xy(self, scale, velocity=False, acceleration=False):
        """
        Sets the x and y components of a length scaled by the amount 'scale'.
        """
        if velocity:
            c_num = self.x_vel + 1j*self.y_vel
            c_num = scale*np.abs(c_num)*np.exp(1j*np.angle(c_num))
            self._x_vel_scaled, self._y_vel_scaled = np.real(c_num), np.imag(c_num)
        elif acceleration:
            c_num = self.x_acc + 1j*self.y_acc
            c_num = scale*np.abs(c_num)*np.exp(1j*np.angle(c_num))
            self._x_acc_scaled, self._y_acc_scaled = np.real(c_num), np.imag(c_num)

    def _get_head_point(self, velocity=False, acceleration=False):
        """
        :return: returns a complex number of a global position of the scaled arrow heads
        """
        Rp = self.x_pos + 1j*self.y_pos
        if velocity:
            R_prime_p = self._x_vel_scaled + 1j*self._y_vel_scaled  # A point relative to the position of the joint
            c_num = Rp + R_prime_p
            return c_num
        elif acceleration:
            R_prime_p = self._x_acc_scaled + 1j*self._y_acc_scaled  # A point relative to the position of the joint
            c_num = Rp + R_prime_p
            return c_num

    def _scale_xys(self, scale, velocity=False, acceleration=False):
        """
        Sets the x and y components of a length scaled by the amount 'scale' for all positions. Used in the animation.
        """
        if velocity:
            c_nums = self.x_velocities + 1j*self.y_velocities
            c_nums = scale*np.abs(c_nums)*np.exp(1j*np.angle(c_nums))
            self._x_vel_scales, self._y_vel_scales = np.real(c_nums), np.imag(c_nums)
        elif acceleration:
            c_nums = self.x_accelerations + 1j*self.y_accelerations
            c_nums = scale*np.abs(c_nums)*np.exp(1j*np.angle(c_nums))
            self._x_acc_scales, self._y_acc_scales = np.real(c_nums), np.imag(c_nums)

    def _get_head_points(self, velocity=False, acceleration=False):
        """
        Populates the head points.
        """
        Rp = self.x_positions + 1j*self.y_positions
        if velocity:
            R_prime_p = self._x_vel_scales + 1j*self._y_vel_scales  # A point relative to the position of the joint
            self._vel_heads = Rp + R_prime_p
        elif acceleration:
            R_prime_p = self._x_acc_scales + 1j*self._y_acc_scales  # A point relative to the position of the joint
            self._acc_heads = Rp + R_prime_p

    def __repr__(self):
        return f'Joint(name={self.name})'

    def __str__(self):
        return self.name


class Mechanism:
    def __init__(self, vectors=None, origin=None, loops=None, pos=None, vel=None, acc=None, guess=None):
        """
        :param vectors: tup, list; A list or tuple of vector objects.
        :param origin: Joint; The joint object to be taken as the origin. This will be assumed to be fixed and forces
                       a fixed frame of reference.
        :param loops: func; This is a function of loop equations of that returns a flattened ndarray. This function is
            used in fsolve. Essentially, a loop is defined as a set of vectors whose sum returns to the original point
            or joint. The set of loop equations will determine how the system moves, and it is the user's responsibility
            to provide an independent/solvable system. This means that the number of unknowns needs to match the number
            of equations, and each loop returns two equations (one for the x direction and one for the y
            direction). Consider an example that contains vectors 'a', 'b', 'c', etc. and has 6 unknowns. If 'x' is an
            array that acts as a means to store the unknowns, and 'i' is the known input to the system,
            the loop equation could look like this:

            def loops(x, i):
                temp = np.zeros((3, 2))  # Three by two matrix of zeros to act as a placeholder.
                temp[0] = a(i) + b(i) - c(x[1]) + d(x[0])
                temp[1] = e(x[2]) + f(x[3]) + d(x[0])
                temp[2] = g(x[4]) - h(x[5]) + i()
                return temp.flatten()

            'a', 'b', 'c', etc. are Vector objects. The index of 'x' corresponds to a system's unknowns, which may be an
            unknown length or angle of a vector. If a vector has an unknown length and angle, then the call signature
            of the vector object would be 'a(r, theta)'. It takes practice to get this right, so it is best to look to
            other examples provided.

        :param pos: int, float, ndarray; Value(s) of pos for the input vector. This gets past as a second argument
            in the loop equation. Could be an angle input or a length input. 
        :param vel: int, float, ndarray; Value(s) of velocity for the input vector. This gets past as a second argument
            in the loop equation when fixing the velocities of the vector objects.
        :param acc: int, float, ndarray; Value(s) of acc for the input vector. This gets past as a second argument
            in the loop equation when fixing the accelerations of the vector objects.
        :param guess: list, tup; List or tuple of ndarrays. The first ndarray is the guess values for position; the
            second is for velocity; the third is for acceleration. Only the position guess is required. If pos, vel,
            and acc are ndarrays, then the guess value corresponds to the first value in the ndarrays.

        Instance Variables
        ------------------
        joints: A list of Joint objects.
        positions: A list of Position objects.
        velocities: A list of Velocity objects.
        accelerations: A list of Acceleration objects.
        """
        self.vectors, self.origin = vectors, origin
        joints = set()
        for v in vectors:
            joints.update(v.joints)
        self.joints = list(joints)

        self.positions, self.velocities, self.accelerations = [], [], []
        for v in self.vectors:
            self.positions.append(v.pos)
            self.velocities.append(v.vel)
            self.accelerations.append(v.acc)

        self.loops = loops
        self.pos = pos  # Angle of the input vector
        self.vel = vel  # Angular velocity of the input vector
        self.acc = acc  # Angular acceleration of the input vector

        self.guess = guess
        self.dic = {v: v for v in self.vectors}

        assert self.vectors, 'Vector argument not defined.'
        assert self.origin, 'Input vector argument not defined.'
        assert self.loops, 'Loops argument not defined.'
        assert self.pos is not None, "pos argument must be defined."

        if isinstance(self.pos, np.ndarray):
            for v in self.vectors:
                v._zero(self.pos.shape[0])

            for j in self.joints:
                j._zero(self.pos.shape[0])

            if isinstance(self.vel, np.ndarray):
                assert self.pos.size == self.vel.size, "vel input size does not match pos input size."

            if isinstance(self.acc, np.ndarray):
                assert self.pos.size == self.acc.size, "acc input size does not match pos input size."

    def _fix_position(self):
        """
        Fixes the positions of all the joints assuming that all vectors are defined locally, meaning that each vector's
        length, angle, r_dot, omega, r_ddot, and alpha are known.
        """
        origin = self.origin
        origin._fix_position(0, 0)

        attached_to_origin = []
        vectors = self.positions[:]

        for v in vectors:
            if v.joints[0] == origin:
                v._fix_global_position()
                attached_to_origin.append(v)
            elif v.joints[1] == origin:
                v_rev = v._reverse()
                v_rev._fix_global_position()
                attached_to_origin.append(v)

        for v in attached_to_origin:
            vectors.remove(v)

        counter = 0
        while not self._position_is_fixed():
            for v in vectors:
                if self._position_is_fixed():
                    break
                for r in attached_to_origin:
                    sum_ = get_sum(r, v)
                    if sum_:
                        attached_to_origin.append(sum_)
                        sum_._fix_global_position()
                        break
            counter += 1
            if counter > 10:
                raise Exception('Not all position vectors are able to be fixed to origin. Are the all joints linked?')

    def _fix_velocity(self):
        """
        Fixes the velocity of all the joints assuming that all vectors are defined locally, meaning that each vector's
        length, angle, r_dot, omega, r_ddot, and alpha are known.
        """
        origin = self.origin
        origin._fix_velocity(0, 0)

        attached_to_origin = []
        vectors = self.velocities[:]

        for v in vectors:
            if v.joints[0] == origin:
                v._fix_global_velocity()
                attached_to_origin.append(v)
            elif v.joints[1] == origin:
                v_rev = v._reverse()
                v_rev._fix_global_velocity()
                attached_to_origin.append(v)

        for v in attached_to_origin:
            vectors.remove(v)

        counter = 0
        while not self._velocity_is_fixed():
            for v in vectors:
                if self._velocity_is_fixed():
                    break
                for r in attached_to_origin:
                    sum_ = get_sum(r, v)
                    if sum_:
                        attached_to_origin.append(sum_)
                        sum_._fix_global_velocity()
                        break
            counter += 1
            if counter > 10:
                raise Exception('Not all velocity vectors are able to be fixed to origin. Are the all joints linked?')

    def _fix_acceleration(self):
        """
        Fixes the accelerations of all the joints assuming that all vectors are defined locally, meaning that the
        vector's length, angle, r_dot, omega, r_ddot, and alpha are known.
        """
        origin = self.origin
        origin._fix_acceleration(0, 0)

        attached_to_origin = []
        vectors = self.accelerations[:]

        for v in vectors:
            if v.joints[0] == origin:
                v._fix_global_acceleration()
                attached_to_origin.append(v)
            elif v.joints[1] == origin:
                v_rev = v._reverse()
                v_rev._fix_global_acceleration()
                attached_to_origin.append(v)

        for v in attached_to_origin:
            vectors.remove(v)

        counter = 0
        while not self._acceleration_is_fixed():
            for v in vectors:
                if self._acceleration_is_fixed():
                    break
                for r in attached_to_origin:
                    sum_ = get_sum(r, v)
                    if sum_:
                        attached_to_origin.append(sum_)
                        sum_._fix_global_acceleration()
                        break
            counter += 1
            if counter > 10:
                raise Exception('Not all velocity vectors are able to be fixed to origin. Are the all joints linked?')

    def _position_is_fixed(self):
        """
        :return: True if all the positions of the joints are fixed.
        """
        for joint in self.joints:
            if not joint._position_is_fixed():
                return False
        return True

    def _velocity_is_fixed(self):
        """
        :return: True if all the velocities of the joints are fixed.
        """
        for joint in self.joints:
            if not joint._velocity_is_fixed():
                return False
        return True

    def _acceleration_is_fixed(self):
        """
        :return: True if all the accelerations of the joints are fixed.
        """
        for joint in self.joints:
            if not joint._acceleration_is_fixed():
                return False
        return True

    def tables(self, position=False, velocity=False, acceleration=False, to_five=False):
        """
        Prints a specified data table.

        :param position: bool; Print position data if set to True
        :param velocity: bool; Print velocity data if set to True
        :param acceleration: bool; Print acceleration data if set to True
        :param to_five: bool; Print all data to five decimal places if set to True.
        """
        if position:
            print('POSITION')
            print('--------\n')
            if not to_five:
                mechanism_data = [[v, v.r, np.rad2deg(v.theta), v.x, v.y] for v in self.positions]
                joint_data = [[j, j.x_pos, j.y_pos] for j in sorted(self.joints, key=lambda x: x.name)]
            else:
                mechanism_data = [[v, f'{v.r:.5f}', f'{np.rad2deg(v.theta):.5f}', f'{v.x:.5f}', f'{v.y:.5f}'] for v
                                  in self.positions]
                joint_data = [[j, f'{j.x_pos:.5f}', f'{j.y_pos:.5f}'] for j in
                              sorted(self.joints, key=lambda x: x.name)]
            Data(mechanism_data, headers=['Vector', 'R', 'Theta', 'x', 'y']).print(table=True)
            print('')
            Data(joint_data, headers=['Joint', 'x', 'y']).print(table=True)
            print('')

        if velocity:
            print('VELOCITY')
            print('--------\n')
            if not to_five:
                mechanism_data = [[v, v._get_mag()[0], np.rad2deg(v._get_mag()[1]), v.x, v.y] for v in
                                  self.velocities]
                omega_slip_data = [[v, v.omega, v.r_dot] for v in self.velocities]
                joint_data = [[j, j._vel_mag()[0], np.rad2deg(j._vel_mag()[1]), j.x_vel, j.y_vel] for j in
                              sorted(self.joints, key=lambda x: x.name)]
            else:
                mechanism_data = [[v, f'{v._get_mag()[0]:.5f}', f'{np.rad2deg(v._get_mag()[1]):.5f}', f'{v.x:.5f}',
                                   f'{v.y:.5f}'] for v in self.velocities]
                omega_slip_data = [[v, f'{v.omega:.5f}', f'{v.r_dot:.5f}'] for v in self.velocities]
                joint_data = [[j, f'{j._vel_mag()[0]:.5f}', f'{np.rad2deg(j._vel_mag()[1]):.5f}', f'{j.x_vel:.5f}',
                               f'{j.y_vel:.5f}'] for j in sorted(self.joints, key=lambda x: x.name)]

            Data(mechanism_data, headers=['Vector', 'Mag', 'Angle', 'x', 'y']).print(table=True)
            print('')
            Data(omega_slip_data, headers=['Vector', 'Omega', 'R_dot']).print(table=True)
            print('')
            Data(joint_data, headers=['Joint', 'Mag', 'Angle', 'x', 'y']).print(table=True)
            print('')

        if acceleration:
            print('ACCELERATION')
            print('------------\n')
            if not to_five:
                mechanism_data = [[v, v._get_mag()[0], np.rad2deg(v._get_mag()[1]), v.x, v.y] for v in
                                  self.accelerations]
                alpha_slip_data = [[v, v.alpha, v.r_ddot] for v in self.accelerations]
                joint_data = [[j, j._acc_mag()[0], np.rad2deg(j._acc_mag()[1]), j.x_acc, j.y_acc] for j in
                              sorted(self.joints, key=lambda x: x.name)]

            else:
                mechanism_data = [
                    [v, f'{v._get_mag()[0]:.5f}', f'{np.rad2deg(v._get_mag()[1]):.5f}', f'{v.x:.5f}', f'{v.y:.5f}'] for v
                    in self.accelerations]
                alpha_slip_data = [[v, f'{v.alpha:.5f}', f'{v.r_ddot:.5f}'] for v in self.accelerations]
                joint_data = [[j, f'{j._acc_mag()[0]:.5f}', f'{np.rad2deg(j._acc_mag()[1]):.5f}', f'{j.x_acc:.5f}',
                               f'{j.y_acc:.5f}'] for j in sorted(self.joints, key=lambda x: x.name)]

            Data(mechanism_data, headers=['Vector', 'Mag', 'Angle', 'x', 'y']).print(table=True)
            print('')
            Data(alpha_slip_data, headers=['Vector', 'Alpha', 'R_ddot']).print(table=True)
            print('')
            Data(joint_data, headers=['Joint', 'Mag', 'Angle', 'x', 'y']).print(table=True)

    def _find_scale(self, x_min, x_max, y_min, y_max, scale_length=0.1, kind='plot', velocity=False,
                    acceleration=False):
        """
        x_min, x_max, y_min, y_max are data points that define the bounding box of the system. The 'kind' parameter
        determine whether to find the scale for an instant "plot" or an "animation" (those are the only two possible
        kinds). The 'velocity' and 'acceleration' parameters determines whether to find the scale for velocity or
        acceleration. The 'scale_length' is the fraction for which the velocity/acceleration vector is to the
        diagonal length of the bounding box.

        Finds the maximum magnitude of velocity/acceleration for all joints, then returns a scale value for which to
        scale up/down
        the velocity arrows when plotting.
        """
        if kind == 'plot':
            if velocity:
                max_mag = max([np.sqrt(j.x_vel**2 + j.y_vel**2) for j in self.joints])
            elif acceleration:
                max_mag = max([np.sqrt(j.x_acc**2 + j.y_acc**2) for j in self.joints])
            else:
                raise Exception('Neither velocity or acceleration specified.')
            # Diagonal of bounding box
            max_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
            return scale_length*max_length/max_mag
        elif kind == 'animation':
            if velocity:
                vel_mags = [j.vel_mags for j in self.joints]
                max_mag = np.amax(vel_mags)
            elif acceleration:
                acc_mags = [j.acc_mags for j in self.joints]
                max_mag = np.amax(acc_mags)
            else:
                raise Exception('Neither velocity or acceleration specified.')
            # Diagonal of bounding box
            max_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
            return scale_length*max_length/max_mag

    def plot(self, velocity=False, acceleration=False, scale=0.1, show_joints=True, grid=True, cushion=1):
        """
        Plots the instance of the mechanism; calculate() method must be called before calling this method.

        :param velocity: bool; Plots velocity vectors if True
        :param acceleration: bool; Plots acceleration vectors if True
        :param scale: float; If velocity or acceleration is specified, the scale will define the relative length of the
                      maximum magnitude to the diagonal of the bounding box. A scale of 0.1 (default) would indicate
                      that the maximum magnitude of the velocity/acceleration is 1/10 the diagonal of the bounding box.
        :param show_joints: bool; Adds joint labels to the plot (only if velocity=False and acceleration=False)
        :param grid: bool; Add the grid if true.
        :param cushion: int, float; The thickness of the cushion around the plot.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        if grid:
            ax.grid(zorder=1)

        y_values = [j.y_pos for j in self.joints]
        x_values = [j.x_pos for j in self.joints]
        min_y, max_y = min(y_values), max(y_values)
        min_x, max_x = min(x_values), max(x_values)

        if velocity:
            assert self.vel is not None, 'There is no input for velocity.'
            sf = self._find_scale(min_x, max_x, min_y, max_y, scale_length=scale, kind='plot', velocity=True)
            for j in self.joints:
                j._scale_xy(sf, velocity=True)
                c_num = j._get_head_point(velocity=True)
                x_values.append(np.real(c_num))
                y_values.append(np.imag(c_num))
        if acceleration:
            assert self.acc is not None, 'There is no input for acceleration.'
            sf = self._find_scale(min_x, max_x, min_y, max_y, scale_length=scale, kind='plot', acceleration=True)
            for j in self.joints:
                j._scale_xy(sf, acceleration=True)
                c_num = j._get_head_point(acceleration=True)
                x_values.append(np.real(c_num))
                y_values.append(np.imag(c_num))

        min_y, max_y = min(y_values), max(y_values)
        min_x, max_x = min(x_values), max(x_values)

        ax.set_xlim(min_x - cushion, max_x + cushion)
        ax.set_ylim(min_y - cushion, max_y + cushion)

        for v in self.positions:
            if not v.show:
                continue
            j1, j2 = v.joints
            v_x = (j1.x_pos, j2.x_pos)
            v_y = (j1.y_pos, j2.y_pos)
            ax.plot(v_x, v_y, **v.kwargs)

        for j in self.joints:
            if velocity:
                c_num = j._get_head_point(velocity=True)
                arrow = FancyArrowPatch(posA=(j.x_pos, j.y_pos), posB=(np.real(c_num), np.imag(c_num)),
                                        **j.vel_arrow_kwargs)
                ax.add_patch(arrow)

            if acceleration:
                c_num = j._get_head_point(acceleration=True)
                arrow = FancyArrowPatch(posA=(j.x_pos, j.y_pos), posB=(np.real(c_num), np.imag(c_num)),
                                        **j.acc_arrow_kwargs)
                ax.add_patch(arrow)
            if not velocity and not acceleration and show_joints:
                ax.annotate(j.name, (j.x_pos, j.y_pos), size='large', zorder=5)

        return fig, ax

    def test(self):
        """
        Checks the distances between joints.
        """
        print('Distances:')
        for v in self.vectors:
            j1, j2 = v.joints
            print(f'- {j1} to {j2}: {np.sqrt((j1.x_pos - j2.x_pos)**2 + (j1.y_pos - j2.y_pos)**2)}')

    def calculate(self):
        """
        Fixes the position of all the joints and vectors. Also fixes the velocity and acceleration data for all the
        vectors and joints if vel and acc for the mechanism is given.
        """
        fsolve(self.loops, self.guess[0], args=(self.pos,))
        self._fix_position()

        if self.vel is not None:
            for v in self.vectors:
                v.get = v.vel.get
                v._update_velocity()

            fsolve(self.loops, self.guess[1], args=(self.vel,))
            self._fix_velocity()

        if self.acc is not None:
            assert self.vel is not None, "vel input not defined, but necessary to solve for accelerations."
            for v in self.vectors:
                v.get = v.acc.get
                v._update_acceleration()

            fsolve(self.loops, self.guess[2], args=(self.acc,))
            self._fix_acceleration()

    def iterate(self):
        """
        Iterates over each pos, vel, and acc input, solving at each instance. Must be called before creating
        an animation. This method must also only be used if pos, vel, and acc are ndarrays. pos argument is a
        minimum requirement.
        """
        assert isinstance(self.pos, np.ndarray), "pos input is not an ndarray."

        guess1 = self.guess[0]
        guess2, guess3 = None, None

        if self.vel is not None:
            guess2 = self.guess[1]

        if self.vel is not None and self.acc is not None:
            guess3 = self.guess[2]

        for i in range(self.pos.shape[0]):
            for v in self.vectors:
                v.get = v.pos.get

            pos = fsolve(self.loops, guess1, args=(self.pos[i],))
            guess1 = pos
            self._fix_position()
            for v in self.vectors:
                v._set_position_data(i)

            for j in self.joints:
                j._set_position_data(i)

            if self.vel is not None:
                for v in self.vectors:
                    v.get = v.vel.get
                    v._update_velocity()

                vel = fsolve(self.loops, guess2, args=(self.vel[i],))
                guess2 = vel
                self._fix_velocity()

                for v in self.vectors:
                    v._set_velocity_data(i)

                for j in self.joints:
                    j._set_velocity_data(i)

            if self.acc is not None:
                assert self.vel is not None, "vel input not defined, but necessary to solve for accelerations."
                for v in self.vectors:
                    v.get = v.acc.get
                    v._update_acceleration()

                acc = fsolve(self.loops, guess3, args=(self.acc[i],))
                guess3 = acc
                self._fix_acceleration()

                for v in self.vectors:
                    v._set_acceleration_data(i)

                for j in self.joints:
                    j._set_acceleration_data(i)

            self.clear_joints()

    def clear_joints(self):
        """
        Clears the joint data. Must be called between two different calls of calculate()
        """
        for joint in self.joints:
            joint._clear()

    def get_bounds(self):
        """
        :return: Two tuples; the first is the minimum and maximum x position of the mechanism, and the second is the
            minimum and maximum y position of the mechanism.
        """
        x_positions = [j.x_positions for j in self.joints]
        y_positions = [j.y_positions for j in self.joints]

        x_min = np.amin(x_positions)
        x_max = np.amax(x_positions)
        y_min = np.amin(y_positions)
        y_max = np.amax(y_positions)

        return (x_min, x_max), (y_min, y_max)

    def get_animation(self, velocity=False, acceleration=False, scale=0.1, stamp=None, stamp_loc=(0.05, 0.9),
                      grid=True, cushion=1, show_joint_names=False):
        """
        :param velocity: bool; Plots velocity vectors if True
        :param acceleration: bool; Plots acceleration vectors if True
        :param scale: float; If velocity or acceleration is specified, the scale will define the relative length of the
                      maximum magnitude to the diagonal of the bounding box. A scale of 0.1 (default) would indicate
                      that the maximum magnitude of the velocity/acceleration is 1/10 the diagonal of the bounding box.
        :param stamp: np.ndarray; Shows a text stamp in the animation for displaying any kind of input. Must be the same
                      size as the input and correspond to the input motion.
        :param stamp_loc: tuple; Position of the stamp in axes transform units. A location of (0.5, 0.75) would place
                          the stamp 50% along the x direction and 75% along the y direction.
        :param grid: bool; Add the grid if true.
        :param cushion: int, float; Add a cushion around the plot.
        :param show_joint_names: bool; Show joint' names if true.
        :return: An animation, figure, and axes object.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        x_limits, y_limits = self.get_bounds()

        vel_arrow_patches, acc_arrow_patches = [], []
        x_points, y_points = [], []
        if velocity:
            assert self.vel is not None, 'There is no input for velocity.'
            sf = self._find_scale(*(x_limits + y_limits), scale_length=scale, kind='animation', velocity=True)
            for j in self.joints:
                j._scale_xys(sf, velocity=True)
                j._get_head_points(velocity=True)
                arrow_obj = FancyArrowPatch(posA=(0, 0), posB=(0, 0), **j.vel_arrow_kwargs)
                vel_arrow_patches.append(arrow_obj)
                ax.add_patch(arrow_obj)
                x_points.extend([j.x_positions, np.real(j._vel_heads)])
                y_points.extend([j.y_positions, np.imag(j._vel_heads)])
        if acceleration:
            assert self.acc is not None, 'There is no input for acceleration.'
            sf = self._find_scale(*(x_limits + y_limits), scale_length=scale, kind='animation', acceleration=True)
            for j in self.joints:
                j._scale_xys(sf, acceleration=True)
                j._get_head_points(acceleration=True)
                arrow_obj = FancyArrowPatch(posA=(0, 0), posB=(0, 0), **j.acc_arrow_kwargs)
                acc_arrow_patches.append(arrow_obj)
                ax.add_patch(arrow_obj)
                x_points.append(np.real(j._acc_heads))
                y_points.append(np.imag(j._acc_heads))

        if grid:
            ax.grid(zorder=1)

        if velocity or acceleration:
            x_min, x_max = np.amin(x_points), np.amax(x_points)
            y_min, y_max = np.amin(y_points), np.amax(y_points)
        else:
            x_min, x_max = x_limits
            y_min, y_max = y_limits
        ax.set_xlim(x_min - cushion, x_max + cushion)
        ax.set_ylim(y_min - cushion, y_max + cushion)

        plot_dict = {}
        
        joints = {}
        """
        a dict to plot names of joints
        """
        for v in self.vectors:
            if not v.pos.show:
                continue

            plot_dict.update({v.pos: ax.plot([], [], **v.pos.kwargs)[0]})
            

        for j in self.joints:
            if j.follow:
                ax.plot(j.x_positions, j.y_positions, **j.kwargs)

        text_list = []
        if stamp is not None:
            assert stamp.size == self.pos.shape[0], "Given stamp array doesn't match the input size."
            text = ax.text(stamp_loc[0], stamp_loc[1], '', transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='white'), zorder=6)
            text_list.append(text)

        def init():
            for vec, line in plot_dict.items():
                j1, j2 = vec.joints
                line.set_data([], [])
                if show_joint_names:
                    xy1 = (j1.x_positions[0], j1.y_positions[0])
                    xy2 = (j2.x_positions[0], j2.y_positions[0])
                    joints.update({j1: plt.annotate(xy=xy1, text=j1.name)})
                    joints.update({j2: plt.annotate(xy=xy2, text=j2.name)})
            for arrow in vel_arrow_patches:
                arrow.set_positions(posA=(0, 0), posB=(0, 0))
            for arrow in acc_arrow_patches:
                arrow.set_positions(posA=(0, 0), posB=(0, 0))
            if text_list:
                text.set_text('')
            return list(plot_dict.values()) + vel_arrow_patches + acc_arrow_patches + text_list

        def animate(i):
            
            for vec, line in plot_dict.items():
                j1, j2 = vec.joints
                line.set_data((j1.x_positions[i], j2.x_positions[i]), (j1.y_positions[i], j2.y_positions[i]))
                if show_joint_names:
                    xy1 = (j1.x_positions[i], j1.y_positions[i])
                    xy2 = (j2.x_positions[i], j2.y_positions[i])
                    joints.update({j1: plt.annotate(xy=xy1, text=j1.name)})
                    joints.update({j2: plt.annotate(xy=xy2, text=j2.name)})
            if velocity:
                for joint, arrow in zip(self.joints, vel_arrow_patches):
                    x_head, y_head = np.real(joint._vel_heads)[i], np.imag(joint._vel_heads)[i]
                    arrow.set_positions(posA=(joint.x_positions[i], joint.y_positions[i]), posB=(x_head, y_head))
            if acceleration:
                for joint, arrow in zip(self.joints, acc_arrow_patches):
                    x_head, y_head = np.real(joint._acc_heads)[i], np.imag(joint._acc_heads)[i]
                    arrow.set_positions(posA=(joint.x_positions[i], joint.y_positions[i]), posB=(x_head, y_head))
            if text_list:
                text.set_text(f'{stamp[i]:.3f}')
            return list(plot_dict.values()) + vel_arrow_patches + acc_arrow_patches + text_list + list(joints.values())

        # noinspection PyTypeChecker
        ani = Player(fig, animate, frames=self.pos.shape[0], interval=50, blit=True, init_func=init)

        return ani, fig, ax

    def __getitem__(self, item):
        return self.dic[item]


def get_joints(names):
    """
    :param names: str; A string with the joint names separated by spaces.
    :return: A list of joint objects.
    """
    return [Joint(ch) for ch in names.split()]


def get_sum(v1, v2):
    """
    This function returns the sum of two vectors. It will reverse the vector(s) in such a way that it only sums the two
    when the head of v1 is attached to the tail of v2.

    :param v1: VectorBase; The vector that is attached to the origin (the tail does not have to be the origin of the
        mechanism).
    :param v2: VectorBase; A vector that has a common joint with v1.
    :return: A VectorBase object sum of v1 and v2. If there are no common joints between the two, then it returns None.
    """
    j1, j2 = v1.joints
    j3, j4 = v2.joints

    if j2 == j3:
        return v1 + v2
    elif j1 == j3:
        return v1._reverse() + v2
    elif j1 == j4:
        return v1._reverse() + v2._reverse()
    elif j2 == j4:
        return v1 + v2._reverse()
    return None


if __name__ == '__main__':
    pass
