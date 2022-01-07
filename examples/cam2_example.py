from mechanism import Cam
import matplotlib.pyplot as plt

cam = Cam(motion=[
    ('Rise', 2, 1.2),
    ('Dwell', 0.3),
    ('Fall', 1, 0.9),
    ('Dwell', 0.6),
    ('Fall', 1, 0.9)
], rotation='cw')

# Showing all the plots for each kind of motion

fig1, ax1 = cam.plot(kind='all')
ax1.grid()

# These return figure and axes objects as well but don't need them.
cam.svaj(kind='naive')
cam.svaj(kind='harmonic')
cam.svaj(kind='cycloidal')

plt.show()

cam.profile(kind='naive', base=2, show_base=True)
cam.profile(kind='harmonic', base=2, show_base=True)
cam.profile(kind='cycloidal', base=2, show_base=True)
fig2, ax2 = cam.profile(kind='all', base=2, show_base=True)
ax2.legend()

plt.show()

ani, fig3, ax3, follower = cam.get_animation(kind='cycloidal', base=2, roller_radius=0.75, width=0.5, length=4, inc=5)
ax3.grid()
plt.show()

# ani.save('../animations/cam2.mp4', dpi=300)

# You can save the coordinates of the cam profile to a file, then use that file to create a curve through xyz points
# in solidworks.
# cam.save_coordinates(file='cam.txt', kind='cycloidal', base=2, solidworks=True)
