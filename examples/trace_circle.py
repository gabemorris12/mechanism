# Just to show that this package has more applications than just mechanisms
# This script will trace a circle
from mechanism import *
import numpy as np
import matplotlib.pyplot as plt

O, A, B = get_joints('O A B')
A.follow = True

a = Vector((O, A), ls=':', color='maroon', marker='o')
b = Vector((B, A), theta=np.pi/2, show=False)
c = Vector((O, B), theta=0, show=False)

r = 2
t = np.linspace(0, 2*np.pi, 250)
points = r*np.exp(1j*t)
x, y = np.real(points) + 5, np.imag(points) + 3
pos = np.concatenate((x.reshape((x.size, 1)), y.reshape((y.size, 1))), axis=1)


def loop(x_, i):
    return a(x_[0], x_[1]) - b(i[1]) - c(i[0])


guess = np.array([3, 3])

mechanism = Mechanism(vectors=[a, b, c], origin=O, loops=loop, pos=pos, guess=(guess, ))
mechanism.iterate()

ani, fig, ax = mechanism.get_animation()
ax.grid(visible=False)

plt.show()
