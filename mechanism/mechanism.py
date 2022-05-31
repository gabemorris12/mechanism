import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

from .dataframe import Data
from .vectors import VectorBase, APPEARANCE


class Joint:
    follow_all = False

    def __init__(self, name='', follow=None, style=None, **kwargs):
        """
        :param: name: str; The name of the joint. Typically, a capital letter.
        :param: follow: bool; If true, the path of the joint will be drawn in the animation.
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

    def position_is_fixed(self):
        """
        :return: True if the position is globally defined.
        """
        return False if self.x_pos is None or self.y_pos is None else True

    def velocity_is_fixed(self):
        """
        :return: True if the velocity is globally defined.
        """
        return False if self.x_vel is None or self.y_vel is None else True

    def acceleration_is_fixed(self):
        """
        :return: True if the acceleration is globally defined.
        """
        return False if self.x_acc is None or self.y_acc is None else True

    def fix_position(self, x_pos, y_pos):
        """
        Sets self.x_pos and self.y_pos
        """
        self.x_pos, self.y_pos = x_pos, y_pos

    def fix_velocity(self, x_vel, y_vel):
        """
        Sets self.x_vel and self.y_vel
        """
        self.x_vel, self.y_vel = x_vel, y_vel
        if abs(self.x_vel) < 1e-10:
            self.x_vel = 0
        if abs(self.y_vel) < 1e-10:
            self.y_vel = 0

    def fix_acceleration(self, x_acc, y_acc):
        """
        Sets self.x_acc and self.y_acc
        """
        self.x_acc, self.y_acc = x_acc, y_acc
        if abs(self.x_acc) < 1e-10:
            self.x_acc = 0
        if abs(self.y_acc) < 1e-10:
            self.y_acc = 0

    def clear(self):
        """
        Clears the non-iterable instance variables. This must be called between two different calls of calculate() from
        the mechanism instance.
        """
        self.x_pos, self.y_pos = None, None
        self.x_vel, self.y_vel = None, None
        self.x_acc, self.y_acc = None, None

    def vel_mag(self):
        """
        :return: A tuple of the magnitude and angle of the velocity of the joint.
        """
        return VectorBase(x=self.x_vel, y=self.y_vel).get_mag()

    def acc_mag(self):
        """
        :return: A tuple of the magnitude and angle of the acceleration of the joint.
        """
        return VectorBase(x=self.x_acc, y=self.y_acc).get_mag()

    def zero(self, s):
        """
        Zeros the iterable instances of the joint object.

        :param s: The size of the iterable instances
        """
        self.x_positions, self.y_positions = np.zeros(s), np.zeros(s)
        self.x_velocities, self.y_velocities = np.zeros(s), np.zeros(s)
        self.x_accelerations, self.y_accelerations = np.zeros(s), np.zeros(s)

        self.vel_mags, self.vel_angles = np.zeros(s), np.zeros(s)
        self.acc_mags, self.acc_angles = np.zeros(s), np.zeros(s)

    def set_position_data(self, i):
        """
        Sets the position data at the index i.

        :param i: Index
        """
        self.x_positions[i] = self.x_pos
        self.y_positions[i] = self.y_pos

    def set_velocity_data(self, i):
        """
        Sets the velocity, vel_mag, and vel_angle data at the index i.

        :param i: Index
        """

        self.x_velocities[i] = self.x_vel
        self.y_velocities[i] = self.y_vel

        mag, angle = self.vel_mag()
        self.vel_mags[i] = mag
        self.vel_angles[i] = angle

    def set_acceleration_data(self, i):
        """
        Sets the acceleration, acc_mag, and acc_angle data at the index i.

        :param i: Index
        """

        self.x_accelerations[i] = self.x_acc
        self.y_accelerations[i] = self.y_acc

        mag, angle = self.acc_mag()
        self.acc_mags[i] = mag
        self.acc_angles[i] = angle

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
            used in fsolve. See examples for how these loop equations are structured.
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
                v.zero(self.pos.shape[0])

            for j in self.joints:
                j.zero(self.pos.shape[0])

            if isinstance(self.vel, np.ndarray):
                assert self.pos.size == self.vel.size, "vel input size does not match pos input size."

            if isinstance(self.acc, np.ndarray):
                assert self.pos.size == self.acc.size, "acc input size does not match pos input size."

    def fix_position(self):
        """
        Fixes the positions of all the joints assuming that all vectors are defined locally, meaning that each vector's
        length, angle, r_dot, omega, r_ddot, and alpha are known.
        """
        origin = self.origin
        origin.fix_position(0, 0)

        attached_to_origin = []
        vectors = self.positions[:]

        for v in vectors:
            if v.joints[0] == origin:
                v.fix_global_position()
                attached_to_origin.append(v)
            elif v.joints[1] == origin:
                v_rev = v.reverse()
                v_rev.fix_global_position()
                attached_to_origin.append(v)

        for v in attached_to_origin:
            vectors.remove(v)

        counter = 0
        while not self.position_is_fixed():
            for v in vectors:
                if self.position_is_fixed():
                    break
                for r in attached_to_origin:
                    sum_ = get_sum(r, v)
                    if sum_:
                        attached_to_origin.append(sum_)
                        sum_.fix_global_position()
                        break
            counter += 1
            if counter > 10:
                raise Exception('Not all position vectors are able to be fixed to origin. Are the all joints linked?')

    def fix_velocity(self):
        """
        Fixes the velocity of all the joints assuming that all vectors are defined locally, meaning that each vector's
        length, angle, r_dot, omega, r_ddot, and alpha are known.
        """
        origin = self.origin
        origin.fix_velocity(0, 0)

        attached_to_origin = []
        vectors = self.velocities[:]

        for v in vectors:
            if v.joints[0] == origin:
                v.fix_global_velocity()
                attached_to_origin.append(v)
            elif v.joints[1] == origin:
                v_rev = v.reverse()
                v_rev.fix_global_velocity()
                attached_to_origin.append(v)

        for v in attached_to_origin:
            vectors.remove(v)

        counter = 0
        while not self.velocity_is_fixed():
            for v in vectors:
                if self.velocity_is_fixed():
                    break
                for r in attached_to_origin:
                    sum_ = get_sum(r, v)
                    if sum_:
                        attached_to_origin.append(sum_)
                        sum_.fix_global_velocity()
                        break
            counter += 1
            if counter > 10:
                raise Exception('Not all velocity vectors are able to be fixed to origin. Are the all joints linked?')

    def fix_acceleration(self):
        """
        Fixes the accelerations of all the joints assuming that all vectors are defined locally, meaning that the
        vector's length, angle, r_dot, omega, r_ddot, and alpha are known.
        """
        origin = self.origin
        origin.fix_acceleration(0, 0)

        attached_to_origin = []
        vectors = self.accelerations[:]

        for v in vectors:
            if v.joints[0] == origin:
                v.fix_global_acceleration()
                attached_to_origin.append(v)
            elif v.joints[1] == origin:
                v_rev = v.reverse()
                v_rev.fix_global_acceleration()
                attached_to_origin.append(v)

        for v in attached_to_origin:
            vectors.remove(v)

        counter = 0
        while not self.acceleration_is_fixed():
            for v in vectors:
                if self.acceleration_is_fixed():
                    break
                for r in attached_to_origin:
                    sum_ = get_sum(r, v)
                    if sum_:
                        attached_to_origin.append(sum_)
                        sum_.fix_global_acceleration()
                        break
            counter += 1
            if counter > 10:
                raise Exception('Not all velocity vectors are able to be fixed to origin. Are the all joints linked?')

    def position_is_fixed(self):
        """
        :return: True if all the positions of the joints are fixed.
        """
        for joint in self.joints:
            if not joint.position_is_fixed():
                return False
        return True

    def velocity_is_fixed(self):
        """
        :return: True if all the velocities of the joints are fixed.
        """
        for joint in self.joints:
            if not joint.velocity_is_fixed():
                return False
        return True

    def acceleration_is_fixed(self):
        """
        :return: True if all the accelerations of the joints are fixed.
        """
        for joint in self.joints:
            if not joint.acceleration_is_fixed():
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
                mechanism_data = [[v, v.get_mag()[0], np.rad2deg(v.get_mag()[1]), v.x, v.y] for v in
                                  self.velocities]
                omega_slip_data = [[v, v.omega, v.r_dot] for v in self.velocities]
                joint_data = [[j, j.vel_mag()[0], np.rad2deg(j.vel_mag()[1]), j.x_vel, j.y_vel] for j in
                              sorted(self.joints, key=lambda x: x.name)]
            else:
                mechanism_data = [[v, f'{v.get_mag()[0]:.5f}', f'{np.rad2deg(v.get_mag()[1]):.5f}', f'{v.x:.5f}',
                                   f'{v.y:.5f}'] for v in self.velocities]
                omega_slip_data = [[v, f'{v.omega:.5f}', f'{v.r_dot:.5f}'] for v in self.velocities]
                joint_data = [[j, f'{j.vel_mag()[0]:.5f}', f'{np.rad2deg(j.vel_mag()[1]):.5f}', f'{j.x_vel:.5f}',
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
                mechanism_data = [[v, v.get_mag()[0], np.rad2deg(v.get_mag()[1]), v.x, v.y] for v in
                                  self.accelerations]
                alpha_slip_data = [[v, v.alpha, v.r_ddot] for v in self.accelerations]
                joint_data = [[j, j.acc_mag()[0], np.rad2deg(j.acc_mag()[1]), j.x_acc, j.y_acc] for j in
                              sorted(self.joints, key=lambda x: x.name)]

            else:
                mechanism_data = [
                    [v, f'{v.get_mag()[0]:.5f}', f'{np.rad2deg(v.get_mag()[1]):.5f}', f'{v.x:.5f}', f'{v.y:.5f}'] for v
                    in self.accelerations]
                alpha_slip_data = [[v, f'{v.alpha:.5f}', f'{v.r_ddot:.5f}'] for v in self.accelerations]
                joint_data = [[j, f'{j.acc_mag()[0]:.5f}', f'{np.rad2deg(j.acc_mag()[1]):.5f}', f'{j.x_acc:.5f}',
                               f'{j.y_acc:.5f}'] for j in sorted(self.joints, key=lambda x: x.name)]

            Data(mechanism_data, headers=['Vector', 'Mag', 'Angle', 'x', 'y']).print(table=True)
            print('')
            Data(alpha_slip_data, headers=['Vector', 'Alpha', 'R_ddot']).print(table=True)
            print('')
            Data(joint_data, headers=['Joint', 'Mag', 'Angle', 'x', 'y']).print(table=True)

    def plot(self, velocity=False, acceleration=False, show_joints=True, grid=True, cushion=1):
        """
        Plots the instance of the mechanism; calculate() method must be called before calling this method.

        :param velocity: bool; Plots velocity vectors if True
        :param acceleration: bool; Plots acceleration vectors if True
        :param show_joints: Adds joint labels to the plot (only if velocity=False and acceleration=False)
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
                ax.quiver(j.x_pos, j.y_pos, j.x_vel, j.y_vel, angles='xy', scale_units='xy', color='deepskyblue',
                          zorder=3)

            if acceleration:
                ax.quiver(j.x_pos, j.y_pos, j.x_acc, j.y_acc, angles='xy', scale_units='xy', color='orange', zorder=3)

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
        self.fix_position()

        if self.vel is not None:
            for v in self.vectors:
                v.get = v.vel.get
                v.update_velocity()

            fsolve(self.loops, self.guess[1], args=(self.vel,))
            self.fix_velocity()

        if self.acc is not None:
            assert self.vel is not None, "vel input not defined, but necessary to solve for accelerations."
            for v in self.vectors:
                v.get = v.acc.get
                v.update_acceleration()

            fsolve(self.loops, self.guess[2], args=(self.acc,))
            self.fix_acceleration()

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
            self.fix_position()
            for v in self.vectors:
                v.set_position_data(i)

            for j in self.joints:
                j.set_position_data(i)

            if self.vel is not None:
                for v in self.vectors:
                    v.get = v.vel.get
                    v.update_velocity()

                vel = fsolve(self.loops, guess2, args=(self.vel[i],))
                guess2 = vel
                self.fix_velocity()

                for v in self.vectors:
                    v.set_velocity_data(i)

                for j in self.joints:
                    j.set_velocity_data(i)

            if self.acc is not None:
                assert self.vel is not None, "vel input not defined, but necessary to solve for accelerations."
                for v in self.vectors:
                    v.get = v.acc.get
                    v.update_acceleration()

                acc = fsolve(self.loops, guess3, args=(self.acc[i],))
                guess3 = acc
                self.fix_acceleration()

                for v in self.vectors:
                    v.set_acceleration_data(i)

                for j in self.joints:
                    j.set_acceleration_data(i)

            self.clear_joints()

    def clear_joints(self):
        """
        Clears the joint data. Must be called between two different calls of calculate()
        """
        for joint in self.joints:
            joint.clear()

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

    def get_animation(self, grid=True, cushion=1):
        # Todo: A step value could be added here to adjust speed
        """
        :param: cushion: int; Add a cushion around the plot.
        :param: grid: bool; Add the grid if true.
        :return: An animation, figure, and axes object.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        x_limits, y_limits = self.get_bounds()

        if grid:
            ax.grid(zorder=1)

        ax.set_xlim(x_limits[0] - cushion, x_limits[1] + cushion)
        ax.set_ylim(y_limits[0] - cushion, y_limits[1] + cushion)

        plot_dict = {}
        for v in self.vectors:
            if not v.pos.show:
                continue

            plot_dict.update({v.pos: ax.plot([], [], **v.pos.kwargs)[0]})

        for j in self.joints:
            if j.follow:
                ax.plot(j.x_positions, j.y_positions, **j.kwargs)

        def init():
            for line in plot_dict.values():
                line.set_data([], [])
            return list(plot_dict.values())

        def animate(i):
            for vec, line in plot_dict.items():
                j1, j2 = vec.joints
                line.set_data((j1.x_positions[i], j2.x_positions[i]), (j1.y_positions[i], j2.y_positions[i]))
            return list(plot_dict.values())

        # noinspection PyTypeChecker
        return FuncAnimation(fig, animate, frames=range(self.pos.shape[0]), interval=50, blit=True,
                             init_func=init), fig, ax

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
        return v1.reverse() + v2
    elif j1 == j4:
        return v1.reverse() + v2.reverse()
    elif j2 == j4:
        return v1 + v2.reverse()
    return None


if __name__ == '__main__':
    pass
