from mechanism import SpurGear
import matplotlib.pyplot as plt
import numpy as np
from mechanism.gears import _rotate

pd = 24
gear = SpurGear(N=48, pd=pd, a=1/pd, b=1.25/pd)
fig, ax = gear.plot()
ax.set_title('Non-Undercutting Gear')

mesh = _rotate(gear.tooth_profile, np.pi + gear.tooth_thickness/gear.r)
mesh = mesh + 1j*gear.d
other_tooth = _rotate(gear.tooth_profile, -2*np.pi/gear.N)

ax.plot(np.real(mesh), np.imag(mesh), color='darkgrey', zorder=3)
ax.plot(np.real(other_tooth), np.imag(other_tooth), color='#000080', zorder=3)

gear2 = SpurGear(N=18, pd=pd, a=1/pd, b=1.25/pd, ignore_undercut=True)
fig2, ax2 = gear2.plot()
ax2.set_title('Undercutting Gear')

mesh2 = _rotate(gear2.tooth_profile, np.pi + gear2.tooth_thickness/gear2.r)
mesh2 = mesh2 + 1j*gear2.d
other_tooth2 = _rotate(gear2.tooth_profile, -2*np.pi/gear2.N)

ax2.plot(np.real(mesh2), np.imag(mesh2), color='darkgrey', zorder=3)
ax2.plot(np.real(other_tooth2), np.imag(other_tooth2), color='#000080', zorder=3)

plt.show()
