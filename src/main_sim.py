import numpy as np
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

from Robotics.Kinematics.quadratic import Quadratic
from Robotics.Robot.baxterBot import BaxterBot
from Robotics.target import Target
from Robotics.Kinematics.robot_movement import RobotMovement

import time
import math

pr = PyRep()

SCENE_FILE = join(dirname(abspath(__file__)), "Demos/Simulations/baxter_robot_arm.ttt")
pr.launch(SCENE_FILE, headless=False)
pr.start()
pr.step_ui()

# arm = get_arm()
bot = BaxterBot()
target = Target()
camera = VisionSensor("Vision_sensor")
rmove = RobotMovement(bot, target, pr)

target.set_restrictedBoundaries()

# target.set_position([0, 1, 0.1])

# bot.find_jointVelo(target, 40)

################ Reaching ################

# target.random_pos()
# bot.trajetoryNoise(target)
# rmove.stayStill(100)

# for _ in range(10):
#     # print("\n\n\n\n")
#     target.random_pos()
#     bot.resetInitial(pr)
#     rmove.resetCurve()
#     # input("Next step")
#     rmove.stayStill(1)
#     # rmove.moveArmCurved(10)
#     rmove.humanMovement(5)

# for _ in range(5):
#     target.random_pos()
#     # target.set_position([0.45, 0.55, 0.025])
#     bot.resetInitial()
#     bot.stayStill(pr, 1)
#     bot.moveArmCurved(pr, target, 3)

# for _ in range(10):
#     target.random_pos()
#     bot.resetInitial()
#     bot.stayStill(pr, 1)
#     bot.trajetoryNoise(pr, target, 5)

# initialConf = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
# bot.robot.set_joint_positions(initialConf)
# bot.stayStill(pr, 1000)

# i=0
# while i<10e6:
#     print(i)
#     i+=1

# bot.move_inDir(pr, np.array([0, 0, 1]), 20)

# bot.stayStill(pr, 1)
# bot.nana(pr, target)

################ Grasping ################
for _ in range(10):
    # print("\n\n\n\n")
    target.random_pos()
    bot.resetInitial(pr)
    rmove.resetCurve()
    # input("Next step")
    rmove.stayStill(1)
    # rmove.moveArmCurved(10)
    rmove.graspingMovement_linear(5)

pr.stop()
pr.shutdown()
