import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join, abspath

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

pr = PyRep()
plt.ion()

SCENE_FILE = join(dirname(abspath(__file__)), "Demos/Simulations/gripper_only.ttt")
pr.launch(SCENE_FILE, headless=False)
pr.start()
pr.step_ui()

# arm = get_arm()

class myBaxter(Gripper):
    def __init__(self, count=0) -> None:
        super().__init__(count, 'BaxterGripper',
                         ['BaxterGripper_closeJoint'])


# gripper = myBaxter()
# gripper = MicoGripper()
gripper = CustomBaxter()

[j.set_control_loop_enabled(False) for j in gripper.joints]
[j.set_joint_target_velocity(0) for j in gripper.joints]
gripper.set_motor_locked_at_zero_velocity(True)


print(f"Target vel: {gripper.get_joint_target_velocities()}")
print(f"Target pos: {gripper.get_joint_target_positions()}")
print(f"Position: {gripper.get_joint_positions()}")
print(f"Interval: {gripper.get_joint_intervals()[1]}")

gripper.set_joint_positions([-0.06, +0.03])

# gripper.set_joint_positions([-0.03, -0.03], disable_dynamics=True)
# gripper.set_joint_positions([-1], disable_dynamics=True)
# gripper.actuate(1, 0.04)

while True:
    # gripper.set_joint_positions([-0.03, -0.03], disable_dynamics=True)
    # print(f"gripper joint pos: {gripper.get_joint_positions()}")
    # print(gripper.actuate(1, 0.01))

    # print(gripper.get_joint_target_velocities())
    pr.step()

pr.stop()
pr.shutdown()
