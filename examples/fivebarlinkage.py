from mechanism import *
import numpy as np
import matplotlib.pyplot as plt

O, A, B, C, D = get_joints('O A B C D')
# B.follow = True

a = Vector((O, A), r=4)
b = Vector((A, B), r=4)
c = Vector((B, C), r=10)
d = Vector((D, C), r=10)
e = Vector((O, D), r=12, theta=0, style='ground')

theta_a = np.arange(0, 2*np.pi, 0.03)
theta_b = 3*theta_a

guess = np.deg2rad([30, 90])


def loops(x, i):
    return a(i[0]) + b(i[1]) + c(x[0]) - d(x[1]) - e()


mechanism = Mechanism(vectors=(a, b, c, d, e), origin=O, guess=(guess,), pos=np.stack((theta_a, theta_b), axis=1),
                      loops=loops)
mechanism.iterate()
ani, fig, ax = mechanism.get_animation()

ax.set_title('Five Bar Linkage')

plt.show()

# ani.save('../animations/fivebarlinkage.mp4', dpi=300)
