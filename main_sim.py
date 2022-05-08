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

print(target.get_orientation())
print(bot.robot.get_orientation())

# target.set_position([0, 1, 0.1])

# bot.find_jointVelo(target, 40)

# target.random_pos()
# bot.trajetoryNoise(target)
# bot.stayStill(pr, 100)

# for _ in range(10):
#     target.random_pos()
#     bot.resetInitial()
#     bot.stayStill(pr, 1)
#     bot.move_arm(pr, target, 5)

for _ in range(10):
    target.random_pos()
    bot.resetInitial()
    bot.stayStill(pr, 1)
    bot.trajetoryNoise(pr, target, 15)

# initialConf = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
# bot.robot.set_joint_positions(initialConf)
# bot.stayStill(pr, 1000)

# i=0
# while i<10e6:
#     print(i)
#     i+=1

# bot.move_inDir(pr, np.array([0, 0, -1]), 20)

# pr.stop()
pr.shutdown()
