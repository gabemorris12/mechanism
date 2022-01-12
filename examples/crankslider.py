from mechanism import *
import numpy as np
from matplotlib import pyplot as plt

O2, A, B, C, D, E = get_joints('O2 A B C D E')
# C.follow = True
a = Vector((O2, A), r=2)
b = Vector((B, A), r=10)
c = Vector((O2, B), theta=0, style='ground')
d = Vector((A, C), r=4)
e = Vector((O2, E), theta=np.pi/2, style='ground')
f = Vector((C, D), r=8)
x = Vector((E, D), r=3, theta=0, style='ground')
y = Vector((B, C), r=7)


# 0: t3, 1: t4, 2: t5, 3: t6, 4: c, 5: e
def loops(t, inp):
    temp = np.zeros((3, 2))
    temp[0] = a(inp) - b(t[0]) - c(t[4])
    temp[1] = e(t[5]) + x() - f(t[2]) - d(t[1]) - a(inp)
    temp[2] = d(t[1]) - y(t[3]) + b(t[0])
    return temp.flatten()


guess = np.concatenate((np.deg2rad([180, 30, 120, 150]), np.array([5, 5])))

# Testing first iteration:
# mechanism = Mechanism(vectors=(a, b, c, d, e, f, x, y), input_vector=a, loops=loops, pos=0, guess=(guess,))
# mechanism.calculate()
# mechanism.plot()

t2 = np.linspace(0, 6*np.pi, 300)
mechanism = Mechanism(vectors=(a, b, c, d, e, f, x, y), input_vector=a, loops=loops, pos=t2, guess=(guess, ))
mechanism.iterate()
ani, fig, ax = mechanism.get_animation()

ax.set_title('Crank Slider')

plt.show()

# ani.save('../animations/crankslider.mp4', dpi=300)

print(np.max(E.y_positions))
print(np.min(E.y_positions))
print(np.max(E.y_positions) - np.min(E.y_positions))
