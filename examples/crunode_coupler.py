from mechanism import *
import numpy as np
import matplotlib.pyplot as plt

O, A, B, C, D = get_joints('O A B C D')
D.follow = True
B.follow = True
a = Vector((O, A), r=1)
b = Vector((O, C), r=2, theta=np.deg2rad(-70), style='ground')
c = Vector((A, B), r=3)
d = Vector((C, B), r=3.5)
e = Vector((A, D), r=2)
f = Vector((O, D), show=False)


def loops(x, inp):
    temp = np.zeros((2, 2))
    temp[0] = a.get(inp) + c.get(x[0]) - d.get(x[1]) - b.get()
    temp[1] = a.get(inp) + e.get(x[0] - np.deg2rad(45)) - f.get(x[2], x[3])
    return temp.flatten()


t2 = np.linspace(0, 6*np.pi, 300)
guess = np.concatenate((np.deg2rad([50, 120]), np.array([5]), np.deg2rad([50])))
mechanism = Mechanism(vectors=(a, b, c, d, e, f), input_vector=a, pos=t2, guess=(guess, ),
                      loops=loops)

mechanism.iterate()
ani, fig, ax = mechanism.get_animation(cushion=1/2)

ax.set_title('Crunode Coupler Curve')

plt.show()

# ani.save('../animations/crunode_coupler.mp4')
