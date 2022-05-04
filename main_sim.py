import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

from my_robot import MyRobot
from target import Target

import time
import math

pr = PyRep()
plt.ion()

SCENE_FILE = join(dirname(abspath(__file__)), "Simulations/coppelia_robot_arm.ttt")
pr.launch(SCENE_FILE, headless=False)
pr.start()
pr.step_ui()

# arm = get_arm()
bot = MyRobot()
target = Target()
camera = VisionSensor("Vision_sensor")

# target.set_position([0, 1, 0.1])

# bot.find_jointVelo(target, 40)
for i in range(40):
    bot.move_arm(pr, target, 50, 5)
# bot.nana()

# i=0
# while i<10e6:
#     print(i)
#     i+=1

# bot.move_inDir(pr, np.array([0, 0, -1]), 20)

# pr.stop()
pr.shutdown()
