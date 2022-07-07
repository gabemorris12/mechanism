from mechanism import SpurGear
import matplotlib.pyplot as plt
import numpy as np

external_gear = SpurGear(N=56, pd=24, agma=True)
fig1, ax1 = external_gear.plot()
ax1.set_title('External Gear Profile')
external_gear.rundown()
print()
external_gear.save_coordinates('external_gear.txt', solidworks=True)

# Should Output
# Property                 | Value
# -------------------------+---------
# Number of Teeth (N)      | 56
# Diametral Pitch (pd)     | 24.00000
# Pitch Diameter (d)       | 2.33333
# Pitch Radius (r)         | 1.16667
# Pressure Angle (phi)     | 20.00000
# Base Radius              | 1.09631
# Addendum (a)             | 0.04167
# Dedendum (b)             | 0.05208
# Circular Tooth Thickness | 0.06462
# Circular Space Width     | 0.06628
# Circular Backlash        | 0.00167

internal_gear = SpurGear(N=56, pd=24, agma=True, internal=True)
fig2, ax2 = internal_gear.plot()
ax2.set_title('Internal Gear Profile')
internal_gear.rundown()
internal_gear.save_coordinates('internal_gear.txt', solidworks=True)

# Should Output
# Property                 | Value
# -------------------------+---------
# Number of Teeth (N)      | 56
# Diametral Pitch (pd)     | 24.00000
# Pitch Diameter (d)       | 2.33333
# Pitch Radius (r)         | 1.16667
# Pressure Angle (phi)     | 20.00000
# Base Radius              | 1.09631
# Addendum (a)             | 0.04167
# Dedendum (b)             | 0.05208
# Circular Tooth Thickness | 0.06628
# Circular Space Width     | 0.06462
# Circular Backlash        | 0.00167

fig3, ax3 = plt.subplots()
ax3.set_title('Mesh')
ax3.plot(np.real(external_gear.tooth_profile), np.imag(external_gear.tooth_profile), color='maroon', label='External')
ax3.plot(np.real(internal_gear.tooth_profile), np.imag(internal_gear.tooth_profile), color='darkgrey', label='Internal')
ax3.legend()
ax3.grid()
ax3.set_aspect('equal')

plt.show()
