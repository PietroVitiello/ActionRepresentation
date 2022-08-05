import numpy as np
from os.path import dirname, join, abspath

from PIL import Image, ImageOps
from torchvision import transforms as T
import torchvision.transforms.functional as ttf
import torch
from torch.utils.data import DataLoader

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.gripper import Gripper
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

from Robotics.Kinematics.quadratic import Quadratic
from Robotics.Robot.my_robot import MyRobot
from Robotics.Robot.custom_baxter import CustomBaxter
from Robotics.target import Target
from Robotics.Kinematics.robot_movement import RobotMovement

import time
import math

from Learning.TrainLoaders.TL_MI import TL_motionImage
from Learning.utils.motion_image import get_motionImage

pr = PyRep()

SCENE_FILE = join(dirname(abspath(__file__)), "Demos/Simulations/gripper_only.ttt")
pr.launch(SCENE_FILE, headless=False)
pr.start()
pr.step_ui()

gripper = CustomBaxter()
gripper.set_position([20, 20, 0.5])

## Object Set

# target = Target()
# target.set_position([1.5, 0.3, 0.025])
# target = Target('rectangle', [0.05, 0.12, 0.05])
# target.set_position([1.5, 0.15, 0.025])
# target = Target('rectangle', [0.12, 0.05, 0.05])
# target.set_position([1.5, 0, 0.025])
# target = Target('rectangle', [0.05, 0.05, 0.12])
# target.set_position([1.5, -0.15, 0.06])
# target = Target('cylinder', [0.05, 0.05, 0.12])
# target.set_position([1.5, -0.3, 0.06])

# color = [0.1, 1, 0.1]
# target = Target(color=color)
# target.set_position([1.35, 0.3, 0.025])
# target = Target('rectangle', [0.05, 0.12, 0.05], color=color)
# target.set_position([1.35, 0.15, 0.025])
# target = Target('rectangle', [0.12, 0.05, 0.05], color=color)
# target.set_position([1.35, 0, 0.025])
# target = Target('rectangle', [0.05, 0.05, 0.12], color=color)
# target.set_position([1.35, -0.15, 0.06])
# target = Target('cylinder', [0.05, 0.05, 0.12], color=color)
# target.set_position([1.35, -0.3, 0.06])

# color = [0.1, 0.1, 1]
# target = Target(color=color)
# target.set_position([1.2, 0.3, 0.025])
# target = Target('rectangle', [0.05, 0.12, 0.05], color=color)
# target.set_position([1.2, 0.15, 0.025])
# target = Target('rectangle', [0.12, 0.05, 0.05], color=color)
# target.set_position([1.2, 0, 0.025])
# target = Target('rectangle', [0.05, 0.05, 0.12], color=color)
# target.set_position([1.2, -0.15, 0.06])
# target = Target('cylinder', [0.05, 0.05, 0.12], color=color)
# target.set_position([1.2, -0.3, 0.06])

# color = [1, 244/255, 0]
# target = Target(color=color)
# target.set_position([1.05, 0.3, 0.025])
# target = Target('rectangle', [0.05, 0.12, 0.05], color=color)
# target.set_position([1.05, 0.15, 0.025])
# target = Target('rectangle', [0.12, 0.05, 0.05], color=color)
# target.set_position([1.05, 0, 0.025])
# target = Target('rectangle', [0.05, 0.05, 0.12], color=color)
# target.set_position([1.05, -0.15, 0.06])
# target = Target('cylinder', [0.05, 0.05, 0.12], color=color)
# target.set_position([1.05, -0.3, 0.06])

## Cube dimensions
target = Target(size=[0.05]*3)
target.set_position([1.5, 0.3, 0.025])
target = Target(size=[0.06]*3)
target.set_position([1.35, 0.3, 0.03])
target = Target(size=[0.07]*3)
target.set_position([1.2, 0.3, 0.035])

for _ in range(20000):
    pr.step()

pr.stop()
pr.shutdown()
