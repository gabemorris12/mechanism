from mechanism import Gear

gear = Gear(N=60, pd=32, agma=True, size=500)
gear.plot(save='../images/gear60.PNG', dpi=240)
gear.save_coordinates(file='gear_tooth_coordinates.txt', solidworks=True)
gear.rundown()
