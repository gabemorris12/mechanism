import numpy as np
from mechanism import Cam
import matplotlib.pyplot as plt

cam = Cam(motion=[
    ('Dwell', 90),
    ('Rise', 1, 90),
    ('Dwell', 90),
    ('Fall', 1, 90)
], degrees=True, omega=2*np.pi)

fig1, ax1 = cam.plot(kind='all')
fig2, ax2 = cam.svaj(kind='cycloidal')
plt.show()

roller_analysis = cam.get_base_circle(kind='cycloidal', follower='roller', roller_radius=1/2, max_pressure_angle=30,
                                      plot=True)
fig3, ax3 = cam.profile(kind='cycloidal', base=roller_analysis['Rb'], show_base=True, roller_radius=1/2,
                        show_pitch=True)
plt.show()

flat_analysis = cam.get_base_circle(kind='cycloidal', follower='flat', desired_min_rho=0.25)
print(flat_analysis['Rb'])
print(flat_analysis['Min Face Width'])
fig4, ax4 = cam.profile(kind='cycloidal', base=flat_analysis['Rb'], show_base=True)
plt.show()

ani, fig5, ax5, follower = cam.get_animation(kind='cycloidal', base=roller_analysis['Rb'], roller_radius=1/2, length=2,
                                             width=3/8, inc=5)
fig6, ax6 = follower.plot()
plt.show()

ani_flat, fig7, ax7, follower = cam.get_animation(kind='cycloidal', base=flat_analysis['Rb'], face_width=2.75, length=2,
                                                  width=3/8, inc=5)
fig8, ax8 = follower.plot()
plt.show()

# cam.save_coordinates('cam_coordinates.txt', kind='cycloidal', base=1.3, solidworks=True)
