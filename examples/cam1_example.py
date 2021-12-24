import numpy as np
from mechanism import Cam

cam = Cam(motion=[
    ('Rise', 3, 90),
    ('Dwell', 15),
    ('Fall', 3, 90),
    ('Dwell', 15),
    ('Rise', 2, 60),
    ('Dwell', 15),
    ('Fall', 2, 60),
    ('Dwell', 15)
], degrees=True, omega=np.pi, rotation='cw')

# Size the cam if using cycloidal motion with a roller follower

roller_analysis = cam.get_base_circle(kind='cycloidal', follower='roller', roller_radius=1, max_pressure_angle=30,
                                      plot=True)
cam.profile(kind='cycloidal', base=roller_analysis['Rb'], show_base=True, show_pitch=True, roller_radius=1)

ani, follower = cam.get_animation(kind='cycloidal', base=4, inc=5, roller_radius=1)

# ani.save('../animations/cam1_roller.mp4', dpi=300)

follower.plot()

ani_, follower = cam.get_animation(kind='harmonic', base=4, inc=5, face_width=4)

# ani_.save('../animations/cam1_flat_face.mp4')

follower.plot()
