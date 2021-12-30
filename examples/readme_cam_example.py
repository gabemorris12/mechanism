import numpy as np
from mechanism import Cam

cam = Cam(motion=[
    ('Dwell', 90),
    ('Rise', 1, 90),
    ('Dwell', 90),
    ('Fall', 1, 90)
], degrees=True, omega=2*np.pi)

cam.plot(kind='all')
cam.svaj(kind='cycloidal')

roller_analysis = cam.get_base_circle(kind='cycloidal', follower='roller', roller_radius=1/2, max_pressure_angle=30,
                                      plot=True)
cam.profile(kind='cycloidal', base=roller_analysis['Rb'], show_base=True, roller_radius=1/2, show_pitch=True,
            loc='best')

flat_analysis = cam.get_base_circle(kind='cycloidal', follower='flat', desired_min_rho=0.25)
print(flat_analysis['Rb'])
print(flat_analysis['Min Face Width'])
cam.profile(kind='cycloidal', base=flat_analysis['Rb'], show_base=True)

ani, follower = cam.get_animation(kind='cycloidal', base=roller_analysis['Rb'], roller_radius=1/2, length=2, width=3/8,
                                  inc=5)
follower.plot()

ani_flat, follower = cam.get_animation(kind='cycloidal', base=flat_analysis['Rb'], face_width=2.75, length=2, width=3/8,
                                       inc=5)
follower.plot()

# cam.save_coordinates('cam_coordinates.txt', kind='cycloidal', base=1.3, solidworks=True)
