from mechanism import Vector, get_joints, Mechanism
import numpy as np
import matplotlib.pyplot as plt

O2, O4, O6, A, B, C, D, E, F, G = get_joints('O2 O4 O6 A B C D E F G')
a = Vector((O4, B), r=2.5)
b = Vector((B, A), r=8.4)
c = Vector((O4, O2), r=12.5, theta=0, style='ground')
d = Vector((O2, A), r=5)
e = Vector((C, A), r=2.4, show=False)
f = Vector((C, E), r=8.9)
g = Vector((O6, E), r=3.2)
h = Vector((O2, O6), r=10.5, theta=np.pi/2, style='ground')
i = Vector((D, E), r=3, show=False)
j = Vector((D, F), r=6.4)
k = Vector((G, F), theta=np.deg2rad(150), style='dotted')
l = Vector((O6, G), r=1.2, theta=np.pi/2, style='dotted')

guess1 = np.concatenate((np.deg2rad([120, 20, 70, 170, 120]), np.array([7])))
guess2 = np.array([15, 15, 30, 12, 30, 3])
guess3 = np.array([10, 10, 30, -30, 20, 10])


def loops(x, inp):
    temp = np.zeros((3, 2))
    temp[0] = a(inp) + b(x[1]) - d(x[0]) - c()
    temp[1] = f(x[2]) - g(x[3]) - h() + d(x[0]) - e(x[1])
    temp[2] = j(x[4]) - k(x[5]) - l() + g(x[3]) - i(x[2])
    return temp.flatten()


mechanism = Mechanism(vectors=(a, b, c, d, e, f, g, h, i, j, k, l), origin=O4, loops=loops,
                      pos=np.deg2rad(52.92024014972946), vel=-30, acc=0, guess=(guess1, guess2, guess3))

mechanism.calculate()
mechanism.tables(acceleration=True, velocity=True, position=True)
fig1, ax1 = mechanism.plot(cushion=2, show_joints=True)
fig2, ax2 = mechanism.plot(cushion=2, velocity=True, acceleration=True)
ax2.set_title('Showing Velocity and Acceleration')

O, A, B, C, P = get_joints('O A B C P')
a = Vector((O, A), r=2)
b = Vector((A, B), r=4.1)
c = Vector((C, B), r=3)
d = Vector((O, C), r=4, theta=0, style='ground')
e = Vector((A, P), r=2.5)
f = Vector((O, P), show=False)


def loops(x, i_):
    temp = np.zeros((2, 2))
    temp[0] = a(i_) + b(x[0]) - c(x[1]) - d()
    temp[1] = a(i_) + e(x[0] + np.deg2rad(30)) - f(x[2], x[3])
    return temp.flatten()


guess1 = np.concatenate((np.deg2rad([20, 60]), np.array([4]), np.deg2rad([48])))
mechanism = Mechanism(vectors=(a, c, b, d, e, f), origin=O, pos=np.deg2rad(45), guess=(guess1, ),
                      loops=loops)
mechanism.calculate()
mechanism.tables(position=True)
fig3, ax3 = mechanism.plot()

plt.show()
