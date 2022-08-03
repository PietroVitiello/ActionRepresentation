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

# data_folder = "linearGrasp_1"
# data_folder = f"Demos/Dataset/{data_folder}/"

# mean = torch.Tensor([0.485, 0.456, 0.406])
# std = torch.Tensor([0.229, 0.224, 0.225])
# transform = T.Compose(
#             [
#                 T.ToTensor(),
#                 T.Normalize(mean.tolist(), std.tolist())
#             ]
#         )

# data = TL_motionImage(data_folder, transform, delta_steps=5, filter_stop=True)
# print(len(data))
# data = DataLoader(data, 64, shuffle=True)
# # mis = []
# for batch, nana in enumerate(data):
#     print(batch * 64)
    # mis.append(nana[0][-1])

# ttf.to_pil_image(mis[10]).show()

frame0 = Image.open("Demos/Dataset/linearGrasp_1/images/episode_0/run_0/step_40.jpg")
frame1 = Image.open("Demos/Dataset/linearGrasp_1/images/episode_0/run_0/step_45.jpg")

frame0.show()
frame1.show()

im = get_motionImage(frame0, frame1, resized_side=32, mi_threshold=150)
# im = T.Resize((32, 32))(im)
im = T.ToTensor()(im)
im = T.ToPILImage()(im)
im.show()

# frame0 = T.Resize((32, 32))(frame0)
# frame1 = T.Resize((32, 32))(frame1)

# frame0.show()
# frame1.show()

# frame0 = np.asarray(frame0, dtype=np.int16)
# frame1 = np.asarray(frame1, dtype=np.int16)

# motion_image = frame1 - frame0
# motion_image *= 5
# motion_image[motion_image<100] = 0
# motion_image[motion_image>255] = 255
# # motion_image[:,:,[1,0]] = 0
# # motion_image[motion_image<200] = 0
# motion_image = motion_image.astype(np.uint8)
# motion_image = Image.fromarray(motion_image[:,:,:])
# motion_image.show()







# pr = PyRep()

# SCENE_FILE = join(dirname(abspath(__file__)), "Demos/Simulations/gripper_only.ttt")
# pr.launch(SCENE_FILE, headless=False)
# pr.start()
# pr.step_ui()

# # arm = get_arm()

# class myBaxter(Gripper):
#     def __init__(self, count=0) -> None:
#         super().__init__(count, 'BaxterGripper',
#                          ['BaxterGripper_closeJoint'])


# # gripper = myBaxter()
# # gripper = MicoGripper()
# gripper = CustomBaxter()

# [j.set_control_loop_enabled(False) for j in gripper.joints]
# [j.set_joint_target_velocity(0) for j in gripper.joints]
# gripper.set_motor_locked_at_zero_velocity(True)


# print(f"Target vel: {gripper.get_joint_target_velocities()}")
# print(f"Target pos: {gripper.get_joint_target_positions()}")
# print(f"Position: {gripper.get_joint_positions()}")
# print(f"Interval: {gripper.get_joint_intervals()[1]}")

# gripper.set_joint_positions([-0.06, +0.03])

# # gripper.set_joint_positions([-0.03, -0.03], disable_dynamics=True)
# # gripper.set_joint_positions([-1], disable_dynamics=True)
# # gripper.actuate(1, 0.04)

# # while True:
# #     # gripper.set_joint_positions([-0.03, -0.03], disable_dynamics=True)
# #     # print(f"gripper joint pos: {gripper.get_joint_positions()}")
# #     # print(gripper.actuate(1, 0.01))

# #     # print(gripper.get_joint_target_velocities())
# #     pr.step()

# pr.stop()
# pr.shutdown()
