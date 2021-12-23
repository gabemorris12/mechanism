from mechanism import *
import numpy as np
import matplotlib.pyplot as plt

O, A, B, C, D = get_joints('O A B C D')
D.follow = True
B.follow = True
a = Vector((O, A), r=5)
# b = Vector((O, C), r=8, theta=np.deg2rad(90), style='ground')  # Use this vector for a cusp output
b = Vector((O, C), r=8, theta=0, style='ground')
c = Vector((A, B), r=8)
d = Vector((C, B), r=9)
e = Vector((A, D), r=4)
f = Vector((O, D), show=False)


def loops(x, inp):
    temp = np.zeros((2, 2))
    temp[0] = a(inp) + c(x[0]) - d(x[1]) - b()
    temp[1] = a(inp) + e(x[0] + np.deg2rad(30)) - f(x[2], x[3])
    return temp.flatten()


t2 = np.linspace(0, 6*np.pi, 300)
guess = np.concatenate((np.deg2rad([50, 120]), np.array([5]), np.deg2rad([50])))
mechanism = Mechanism(vectors=(a, b, c, d, e, f), input_vector=a, pos=t2, guess=(guess, ),
                      loops=loops)

mechanism.iterate()
ani = mechanism.get_animation()
plt.show()

# ani.save('../animations/fourbarlinkage.mp4', dpi=300)
