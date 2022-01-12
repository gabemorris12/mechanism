from mechanism import SpurGear
import matplotlib.pyplot as plt

gear = SpurGear(N=60, pd=32, agma=True, size=500)
fig, ax = gear.plot()
fig.savefig('../images/gear60.PNG', dpi=240)
plt.show()
gear.save_coordinates(file='gear_tooth_coordinates.txt', solidworks=True)
gear.rundown()
