from mechanism import *
import numpy as np
import matplotlib.pyplot as plt

# Define joints first
O, A, B, P = get_joints('O A B P')
B.follow = True

# Define vectors
# We omit the arguments if they are values that change. If the parameter is constant (such as the crank length), we pass
# the argument.
a = Vector((O, A), r=6)
b = Vector((A, B), r=10)
c = Vector((P, B), r=3, theta=np.pi/2, style='ground')
d = Vector((O, P), theta=0, style='ground')


# Define the loop equation(s)
# This needs to be a function and has to return a flattened array. See other examples that have more than one loop
# equation.
def loop(x, inp):
    # inp is the input and x is an array of unknown values
    return a(inp) + b(x[0]) - c() - d(x[1])  # pass the unknown values into the __call__ method


# Define the known input and the guess values
# The guess values are for the first iteration
t = np.linspace(0, 0.1, 250)  # time
th2 = 20*np.pi*t  # theta2 in rad
w2 = np.full(th2.size, 20*np.pi)  # omega2 in rad/min
a2 = np.zeros(th2.size)  # alpha2 in rad/min^2
# The guess values need to be an array of the same length as x in the loop equation. The order of the guess values is
# also the same as the order defined in the loop equation.
pos_guess = np.array([np.pi/4, 10])
vel_guess = np.array([10, 10])
acc_guess = np.array([10, 10])

# Define the mechanism
mechanism = Mechanism(vectors=[a, b, c, d], origin=O, loops=loop, pos=th2, vel=w2, acc=a2,
                      guess=(pos_guess, vel_guess, acc_guess))
mechanism.iterate()  # solves for each item in the defined input arrays


# Get the animation, figure, and axes objects
ani, fig, ax = mechanism.get_animation()
ax.set_title('Offset Crank Slider')

# ani.save('../animations/offset_crankslider.mp4', dpi=240)

# Plot the velocity of point B
# Access the velocity of point B through r_dot of vector d. This will show the magnitude as well as the direction. The
# velocity may also be accessed through the joint object, B.x_velocities.
fig2, ax2 = plt.subplots()
ax2.plot(t, d.vel.r_dots, color='maroon', label=r'$\dot{R}_d$')
ax2.set_title('Velocity of Joint B')
ax2.set_xlabel('Time (min)')
ax2.set_ylabel(r'Velocity ($\frac{in}{min}$)')
ax2.legend()
ax2.grid()

plt.show()
