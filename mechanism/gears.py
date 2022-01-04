import numpy as np
import matplotlib.pyplot as plt
from .vectors import APPEARANCE
import json


class Gear:

    with open(APPEARANCE, 'r') as f:
        appearance = json.load(f)

    gear_appearance = appearance['gear_plot']

    def __init__(self, N=0, pd=0, d=0, pressure_angle=20, a=0, b=0, tooth_thickness=0):
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

        self.a, self.b = a, b
        self.tooth_thickness = tooth_thickness  # This is the arc length
        self.r = self.d/2
        self.r_base = self.r*np.cos(self.pressure_angle)
        self.ra, self.rb = self.r + self.a, self.r - self.b

        if self.rb >= self.r_base:
            theta_min = np.sqrt((self.rb/self.r_base)**2 - 1)
        else:
            theta_min = 0

        theta_max = np.sqrt((self.ra/self.r_base)**2 - 1)

        theta_pitch = np.sqrt((self.r/self.r_base)**2 - 1)
        x_pitch = self.r_base*np.cos(theta_pitch) + self.r_base*theta_pitch*np.sin(theta_pitch)
        y_pitch = self.r_base*np.sin(theta_pitch) - self.r_base*theta_pitch*np.cos(theta_pitch)
        theta_pitch = np.angle((x_pitch + 1j*y_pitch))  # The angle for the point at the pitch circle

        circular_pitch_angle = self.tooth_thickness/self.r
        rotation = np.pi/2 - theta_pitch - circular_pitch_angle/2

        thetas = np.linspace(theta_min, theta_max, 1000)
        x = self.r_base*np.cos(thetas) + self.r_base*thetas*np.sin(thetas)
        y = self.r_base*np.sin(thetas) - self.r_base*thetas*np.cos(thetas)
        involute_points = x + 1j*y
        # noinspection PyTypeChecker
        involute_points = np.insert(involute_points, 0, self.rb + 0j) if self.rb < self.r_base else involute_points

        self.involute_points = np.abs(involute_points)*np.exp(1j*(np.angle(involute_points) + rotation))
        self.involute_reflection = -1*np.real(self.involute_points) + 1j*np.imag(self.involute_points)
        addendum_start = np.angle(self.involute_points[-1])
        addendum_end = np.angle(self.involute_reflection[-1])
        addendum_circle = self.ra*np.exp(1j*np.arange(addendum_start + 0.006, addendum_end, 0.006))

        self.tooth_profile = np.concatenate((self.involute_points, addendum_circle, np.flip(self.involute_reflection)))

    def plot(self, save='', dpi=240):
        angle_start = np.angle(self.involute_points[0]) - np.deg2rad(5)
        angle_end = np.angle(self.involute_reflection[0]) + np.deg2rad(5)
        thetas = np.linspace(angle_start, angle_end, 1000)

        base = self.r_base*np.exp(1j*thetas)
        pitch = self.r*np.exp(1j*thetas)

        fig, ax = plt.subplots()
        ax.plot(np.real(base), np.imag(base), **Gear.gear_appearance['base'])
        ax.plot(np.real(pitch), np.imag(pitch), **Gear.gear_appearance['pitch'])
        ax.plot(np.real(self.tooth_profile), np.imag(self.tooth_profile), **Gear.gear_appearance['tooth'])

        ax.set_aspect('equal')
        ax.legend()
        ax.grid()

        if save:
            fig.savefig(save, dpi=dpi)

        plt.show()
