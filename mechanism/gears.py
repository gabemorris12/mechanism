import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from .vectors import APPEARANCE


class Gear:
    with open(APPEARANCE, 'r') as f:
        appearance = json.load(f)

    gear_appearance = appearance['gear_plot']

    def __init__(self, N=0, pd=0, d=0, pressure_angle=20, agma=False, a=0, b=0, tooth_thickness=0, size=1000):
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
        else:
            self.a, self.b = a, b
            self.tooth_thickness = tooth_thickness  # This is the arc length (not a straight line)

        self.ra, self.rb = self.r + self.a, self.r - self.b

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
        circular_pitch_angle = self.tooth_thickness/self.r
        rotation = np.pi/2 - theta_pitch - circular_pitch_angle/2

        # Get the involute curve. Create just a linear relationship if the dedendum radius is less than the base radius.
        thetas = np.linspace(theta_min, theta_max, size)
        x = self.r_base*np.cos(thetas) + self.r_base*thetas*np.sin(thetas)
        y = self.r_base*np.sin(thetas) - self.r_base*thetas*np.cos(thetas)
        involute_points = x + 1j*y
        # noinspection PyTypeChecker
        involute_points = np.insert(involute_points, 0, self.rb + 0j) if self.rb < self.r_base else involute_points

        # Rotate the involute curve and get the reflection/addendum circle
        self.involute_points = np.abs(involute_points)*np.exp(1j*(np.angle(involute_points) + rotation))
        self.involute_reflection = -1*np.real(self.involute_points) + 1j*np.imag(self.involute_points)
        addendum_start = np.angle(self.involute_points[-1])
        addendum_end = np.angle(self.involute_reflection[-1])
        addendum_circle = self.ra*np.exp(1j*np.linspace(addendum_start, addendum_end, size))

        # Construct the tooth profile
        self.tooth_profile = np.concatenate((self.involute_points, addendum_circle[1:-1],
                                             np.flip(self.involute_reflection)))

    def plot(self, save='', **kwargs):
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
            fig.savefig(save, **kwargs)

        plt.show()

    def save_coordinates(self, file='', solidworks=False):
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
