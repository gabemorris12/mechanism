# Testing the curve below the base circle. The curve below the base circle is assumed to be a reflected involute.
from mechanism import SpurGear
import numpy as np
import matplotlib.pyplot as plt

# Should see this output
# Property                 | Value
# -------------------------+---------
# Number of Teeth (N)      | 14
# Diametral Pitch (pd)     | 8.00000
# Pitch Diameter (d)       | 1.75000
# Pitch Radius (r)         | 0.87500
# Pressure Angle (phi)     | 20.00000
# Base Radius              | 0.82223
# Addendum (a)             | 0.12500
# Dedendum (b)             | 0.15625
# Circular Tooth Thickness | 0.18408
# Circular Space Width     | 0.20862
# Circular Backlash        | 0.02454

pd = 8
backlash = (1/16)*np.pi/pd
sun = SpurGear(N=14, pd=pd, agma=True, backlash=backlash, size=1000, ignore_undercut=False)
fig1, ax1 = sun.plot()

og = np.genfromtxt('sun_gear_original.csv', delimiter=',')
x, y = og[:, 0], og[:, 1]
ax1.plot(x, y, label='Original', zorder=4, color='darkgrey', ls='-.')
ax1.legend()

sun.rundown()
plt.show()
