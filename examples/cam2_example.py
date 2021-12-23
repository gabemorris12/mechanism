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

cam.plot(kind='all')

cam.svaj(kind='naive')
cam.svaj(kind='harmonic')
cam.svaj(kind='cycloidal')

cam.profile(kind='naive', base=2, show_base=True)
cam.profile(kind='harmonic', base=2, show_base=True)
cam.profile(kind='cycloidal', base=2, show_base=True)
cam.profile(kind='all', base=2, show_base=True, loc='best')

ani, follower = cam.get_animation(kind='cycloidal', base=2, roller_radius=0.75, width=0.5, length=4, inc=5)
plt.show()

# ani.save('../animations/cam2.mp4', dpi=300)

# You can save the coordinates of the cam profile to a file, then use that file to create a curve through xyz points
# in solidworks.
# cam.save_coordinates(file='cam.txt', kind='cycloidal', base=2, solidworks=True)
