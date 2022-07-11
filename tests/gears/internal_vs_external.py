# This is to test a visual that the internal gears are able to nest their external counterpart
from mechanism import SpurGear
import numpy as np
import matplotlib.pyplot as plt

pd = 24
backlash = 0.08/pd

for N in range(30, 56, 5):
    gear_external = SpurGear(N=N, pd=pd, agma=True, ignore_undercut=True, backlash=backlash)
    gear_internal = SpurGear(N=N, pd=pd, agma=True, ignore_undercut=True, backlash=backlash, internal=True)
    fig, ax = gear_internal.plot()
    ax.set_title(f'N={N}')
    ax.plot(np.real(gear_external.tooth_profile), np.imag(gear_external.tooth_profile), ls='-.',
            color='darkgrey', label='External Gear')
    ax.legend()
    plt.show()
