import numpy as np
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APPEARANCE = os.path.join(THIS_DIR, 'appearance.json')


class VectorBase:
    def __init__(self, joints=None, r=None, theta=None, x=None, y=None, show=True, style=None, **kwargs):
        """
        :param joints: tup; A tuple of Joint objects. The first Joint is the tail, and the second is the head.
        :param r: int, float; The length of the vector. If specified, the vector is taken to be a constant length,
            meaning r_dot and r_ddot is zero.
        :param theta: int, float; The angle of the vector in radians from the positive x-axis. Counter clockwise is
            positive. If specified, the vector is directed at a constant angle, meaning omega and alpha is zero.
        :param x: int, float; The value of the x component of the vector.
        :param y: int, float; The value of the y component of the vector.
        :param show: bool; If True, then the vector will be present in plots and animations.
        :param style: str; Applies a certain style passed to plt.plot().
            Options: 
                ground - a dashed black line for grounded link
        :param kwargs: Extra arguments that are passed to plt.plot(). If not specified, the line will be maroon with a
            marker style = 'o'

        Instance Variables
        ------------------
        rs: An ndarray of r values.
        thetas: An ndarray of theta values.
        r_dots: An ndarray of r_dot values (r_dot is the rate of change of the length of the vector length with respect
            to time).
        omegas: An ndarray of omega values (omega is the rate of change of the angle of the vector with respect to
            time).
        r_ddots: An ndarray of r_ddot values (r_ddot is the rate of change of the rate of change of the vector length
            with respect to time).
        alphas: An ndarray of alpha values (alpha is the rate of change of the rate of change of the angle of the vector
            with respect to time).
        get: A function that returns the x and y component of the vector. The arguments for this function depends on
            what is specified at the initialization of the object. For instance, if r is set to a certain value, the get
            function will require an angle input.
        """
        self.joints, self.r, self.theta, self.show = joints, r, theta, show

        with open(APPEARANCE, 'r') as f:
            appearance = json.load(f)

        if style:
            self.kwargs = appearance['mechanism_plot'][style]
        elif kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = appearance['mechanism_plot']['default']

        self.x, self.y = x, y

        self.rs, self.thetas, self.r_dots, self.omegas, self.r_ddots, self.alphas = None, None, None, None, None, None

        if self.r is not None and self.theta is not None:
            self.r_dot, self.omega = 0, 0
            self.r_ddot, self.alpha = 0, 0
            self.get = self.neither
        elif self.r is not None and self.theta is None:
            self.r_dot, self.omega = 0, None
            self.r_ddot, self.alpha = 0, None
            self.get = self.tangent
        elif self.r is None and self.theta is not None:
            self.r_dot, self.omega = None, 0
            self.r_ddot, self.alpha = None, 0
            self.get = self.slip
        else:
            self.r_dot, self.omega = None, None
            self.r_ddot, self.alpha = None, None
            self.get = self.both

    def neither(self):
        pass

    def both(self, _, __):
        pass

    def slip(self, _):
        pass

    def tangent(self, _):
        pass

    def fix_global_position(self):
        """
        Fixes the position of the head joint by making its position the x and y components of the current instance.
        """
        self.joints[1].fix_position(self.x, self.y)

    def fix_global_velocity(self):
        """
        Fixes the velocity of the head joint by making its velocity the x and y components of the current instance.
        """
        self.joints[1].fix_velocity(self.x, self.y)

    def fix_global_acceleration(self):
        """
        Fixes the acceleration of the head joint by making its acceleration the x and y components of the current
        instance.
        """
        self.joints[1].fix_acceleration(self.x, self.y)

    def reverse(self):
        """
        :return: A VectorBase object that is reversed. The joints get reversed as well as the x and y components.
        """
        x, y = -self.x, -self.y
        return VectorBase(joints=(self.joints[1], self.joints[0]), x=x, y=y)

    def get_mag(self):
        """
        :return: A tuple consisting of the magnitude of the current instance and the angle.
        """
        mag = np.sqrt(self.x ** 2 + self.y ** 2)

        if self.x > 0 and self.y >= 0:
            angle = np.arctan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            angle = np.arctan(self.y/self.x) + np.pi
        elif self.x < 0 and self.y <= 0:
            angle = np.arctan(self.y/self.x) + np.pi
        elif self.x > 0 and self.y <= 0:
            angle = np.arctan(self.y/self.x) + 2*np.pi
        elif self.x == 0 and self.y >= 0:
            angle = np.pi/2  # when x is zero
        else:
            angle = 3*np.pi/2

        return mag, angle

    def __add__(self, other):
        x, y = self.x + other.x, self.y + other.y
        return VectorBase(joints=(self.joints[0], other.joints[1]), x=x, y=y)


class Vector:
    def __init__(self, joints=None, r=None, theta=None, x=None, y=None, show=True, style=None, **kwargs):
        """
        See the VectorBase class for details regarding the parameters. The purpose of this class is to group Position,
        Velocity, and Acceleration objects.

        Instance Variables
        ------------------
        pos: Position object which is a subclass of VectorBase. Does not include the r_dot, omega, r_ddot, and alpha
            attributes.
        vel: Velocity object which is a subclass of VectorBase. Does not include the r_ddot and alpha attributes.
        acc: Acceleration object which is a subclass of VectorBase.
        """
        self.pos = Position(joints=joints, r=r, theta=theta, x=x, y=y, show=show, style=style, **kwargs)
        self.vel = Velocity(joints=joints, r=r, theta=theta, x=x, y=y, show=show, style=style, **kwargs)
        self.acc = Acceleration(joints=joints, r=r, theta=theta, x=x, y=y, show=show, style=style, **kwargs)

        self.get = self.pos.get
        self.joints = joints

    def update_velocity(self):
        """
        Updates the velocity object to include the length, r, and the angle ,theta.
        """
        self.vel.r = self.pos.r
        self.vel.theta = self.pos.theta

    def update_acceleration(self):
        """
        Updates the acceleration object to include r, theta, r_dot, and omega.
        """
        self.acc.r = self.vel.r
        self.acc.theta = self.vel.theta
        self.acc.r_dot = self.vel.r_dot
        self.acc.omega = self.vel.omega

    def zero(self, s):
        """
        Zeros all the ndarray attributes at a certain size, s.

        :param s: int; The size of the data
        """
        self.pos.rs = np.zeros(s)
        self.pos.thetas = np.zeros(s)
        self.vel.rs = self.pos.rs
        self.vel.thetas = self.pos.thetas
        self.vel.r_dots = np.zeros(s)
        self.vel.omegas = np.zeros(s)
        self.acc.rs = self.vel.rs
        self.acc.thetas = self.vel.thetas
        self.acc.r_dots = self.vel.r_dots
        self.acc.omegas = self.vel.omegas
        self.acc.r_ddots = np.zeros(s)
        self.acc.alphas = np.zeros(s)

    def set_position_data(self, i):
        """
        Sets position data at index, i.

        :param i: Index
        """
        self.pos.rs[i] = self.pos.r
        self.pos.thetas[i] = self.pos.theta

    def set_velocity_data(self, i):
        """
        Sets velocity data at index, i.

        :param i: Index
        """
        self.vel.r_dots[i] = self.vel.r_dot
        self.vel.omegas[i] = self.vel.omega

    def set_acceleration_data(self, i):
        """
        Sets acceleration data at index, i.

        :param i: Index
        """
        self.acc.r_ddots[i] = self.acc.r_ddot
        self.acc.alphas[i] = self.acc.alpha

    def __call__(self, *args):
        return self.get(*args)

    def __repr__(self):
        return f'{self.joints[0]}{self.joints[1]}'


class Position(VectorBase):
    def __init__(self, **kwargs):
        VectorBase.__init__(self, **kwargs)
        del self.r_dot, self.r_ddot, self.omega, self.alpha, self.r_dots, self.omegas, self.r_ddots, self.alphas

    def both(self, r, theta):
        # When both the theta and the radius is unknown
        self.x, self.y = r*np.cos(theta), r*np.sin(theta)
        self.r, self.theta = r, theta
        return np.array([self.x, self.y])

    def neither(self):
        self.x, self.y = self.r*np.cos(self.theta), self.r*np.sin(self.theta)
        return np.array([self.x, self.y])

    def tangent(self, theta):
        # When the theta is unknown
        self.x, self.y = self.r*np.cos(theta), self.r*np.sin(theta)
        self.theta = theta
        return np.array([self.x, self.y])

    def slip(self, r):
        # When the radius is unknown
        self.x, self.y = r*np.cos(self.theta), r*np.sin(self.theta)
        self.r = r
        return np.array([self.x, self.y])

    def __repr__(self):
        return f'Position(joints={self.joints}, r={self.r}, theta={self.theta})'

    def __str__(self):
        return f'R_{self.joints[0]}{self.joints[1]}'


class Velocity(VectorBase):
    def __init__(self, **kwargs):
        VectorBase.__init__(self, **kwargs)
        del self.r_ddot, self.alpha, self.r_ddots, self.alphas

    def neither(self):
        self.x, self.y = 0, 0
        return np.array([self.x, self.y])

    def both(self, r_dot, omega):
        self.x, self.y = (r_dot*np.cos(self.theta) - self.r*omega*np.sin(self.theta),
                          r_dot*np.sin(self.theta) + self.r*omega*np.cos(self.theta))
        self.r_dot, self.omega = r_dot, omega
        return np.array([self.x, self.y])

    def tangent(self, omega):
        self.x, self.y = self.r*omega*-np.sin(self.theta), self.r*omega*np.cos(self.theta)
        self.omega = omega
        return np.array([self.x, self.y])

    def slip(self, r_dot):
        self.x, self.y = r_dot*np.cos(self.theta), r_dot*np.sin(self.theta)
        self.r_dot = r_dot
        return np.array([self.x, self.y])

    def __repr__(self):
        return f'Velocity(joints={self.joints}, r_dot={self.r_dot}, omega={self.omega})'

    def __str__(self):
        return f'V_{self.joints[0]}{self.joints[1]}'


class Acceleration(VectorBase):
    def __init__(self, **kwargs):
        VectorBase.__init__(self, **kwargs)

    def neither(self):
        self.x, self.y = 0, 0
        return np.array([self.x, self.y])

    def both(self, r_ddot, alpha):
        self.x, self.y = (r_ddot*np.cos(self.theta) - 2*self.r_dot*self.omega*np.sin(self.theta) - self.r*alpha*np.sin(
            self.theta) - self.r*self.omega ** 2*np.cos(self.theta),
                          r_ddot*np.sin(self.theta) + 2*self.r_dot*self.omega*np.cos(self.theta) + self.r*alpha*np.cos(
                              self.theta) - self.r*self.omega ** 2*np.sin(self.theta))
        self.r_ddot, self.alpha = r_ddot, alpha
        return np.array([self.x, self.y])

    def tangent(self, alpha):
        self.x, self.y = (self.r*alpha*-np.sin(self.theta) - self.r*self.omega**2*np.cos(self.theta),
                          self.r*alpha*np.cos(self.theta) - self.r*self.omega**2*np.sin(self.theta))
        self.alpha = alpha
        return np.array([self.x, self.y])

    def slip(self, r_ddot):
        self.x, self.y = r_ddot*np.cos(self.theta), r_ddot*np.sin(self.theta)
        self.r_ddot = r_ddot
        return np.array([self.x, self.y])

    def __repr__(self):
        return f'Acceleration(joints={self.joints}, r_ddot={self.r_ddot}, alpha={self.alpha})'

    def __str__(self):
        return f'A_{self.joints[0]}{self.joints[1]}'
