# My project was to create a table with a mechanical leaf that rises up.

from mechanism import *
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Parameters (want all distances in feet)
inches = 1/12

l = 24*inches  # Leaf width
p = 1/4*inches  # pitch
omega = 300  # rpm of the lead screw
a = 3  # half the length of the table in position 1
b1 = 1.5*inches  # bracket 1 depth
w1 = 4*inches  # bracket 1 distance from edge
b2 = 3*inches  # bracket 2 depth
w2 = 1*inches  # bracket 2 distance from edge
s = 0.25*inches  # Space parameter for when the leaf is in position 2

# Calculating h and m
m_xf = w2 + a + s - w1
m_yf = b2 - b1
m = np.sqrt(m_xf**2 + m_yf**2)
m_xi = a + w2 - w1 - l/2
h = np.sqrt(m**2 - m_xi**2) + b1 - b2
ti = np.arccos(m_xi/m)
assert ti < np.deg2rad(60), 'Initial angle of connecting rod too large. Too much vertical force.'

print(f'h: {h}')
print(f'm: {m}')
print(f'ti: {np.rad2deg(ti)}')

# Setting up input
v_leaf = 1/60*(omega*p)
t = h/v_leaf
time = np.linspace(0, t, 150)
v1_pos = v_leaf*time
v1_vel = v_leaf*np.ones(time.size)
v1_acc = np.zeros(time.size)
print(f't: {time}')

# Defining vectors
# noinspection DuplicatedCode
O, B, C, D, E, F, G, H, I, L, M = get_joints('O B C D E F G H I L M')
v1 = Vector((O, B), theta=np.pi/2, style='ground')
v2 = Vector((B, C), r=l/2 - w2, theta=0, color='maroon', lw=2)
v3 = Vector((C, D), r=b2, theta=-np.pi/2, style='ground')
v4 = Vector((D, E), r=m, color='maroon', lw=2)
v5 = Vector((F, E), r=h - b1, theta=np.pi/2, style='ground')
v6 = Vector((O, F), theta=0, style='ground')
v7 = Vector((B, G), theta=np.pi/2, show=False)
v8 = Vector((G, H), theta=0, style='ground')
v9 = Vector((H, I), r=a - w1, theta=0, color='maroon', lw=2)
v10 = Vector((E, I), r=b1, theta=np.pi/2, style='ground')
v11 = Vector((C, L), r=w2, theta=0, color='maroon', lw=2)
v12 = Vector((L, D), show=False)
v13 = Vector((I, M), r=w1, theta=0, color='maroon', lw=2)
v14 = Vector((E, M), show=False)


def loops(x, inp):
    temp = np.zeros((4, 2))
    temp[0] = v1(inp) + v2() + v3() + v4(x[0]) - v5() - v6(x[1])
    temp[1] = v7(x[2]) + v8(x[3]) + v9() - v10() - v4(x[0]) - v3() - v2()
    temp[2] = v11() + v12(x[4], x[5]) - v3()
    temp[3] = v13() - v14(x[6], x[7]) + v10()
    return temp.flatten()


# Guesses
pos_guess = [0.6591975977642502, 3.5000000000122213, 2.111989978719915, 1.2221305849199858e-11, 1*inches,
             np.deg2rad(-120), 1*inches, np.deg2rad(45)]
vel_guess = [-0.019607843137219604, 0.043045555138092384, -0.05555555555555556, 0.043045555138092384,
             0.043045555138092384, 0.043045555138092384, 0.043045555138092384, 0.043045555138092384]
acc_guess = [0.00029789311514143526, -0.0017432963189026164, -2.213891507590676e-25, -0.0017432963189026164,
             0.043045555138092384, 0.043045555138092384, 0.043045555138092384, 0.043045555138092384]

table = Mechanism(vectors=[eval(f'v{i}') for i in range(1, 15)], input_vector=v1, loops=loops, pos=v1_pos,
                  vel=v1_vel, acc=v1_acc, guess=[pos_guess, vel_guess, acc_guess])
table.iterate()
ani, fig, ax = table.get_animation(cushion=0.25)

ax.set_title('Extendable Table')
ax.grid()

plt.show()

# Show the velocity of the leaf and table tops.
plt.plot(time, v1.acc.r_ddots, color='black', label='Slip $A_1$')
plt.plot(time, v8.acc.r_ddots, color='maroon', label='Slip $A_8$')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.title('Acceleration of Leaf and Tabletop')
plt.show()

# ani.save('../animations/table.mp4', dpi=300)
