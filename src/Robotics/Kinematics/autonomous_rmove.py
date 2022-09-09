import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape, Dummy

from .quadratic import Quadratic
from ..Robot.my_robot import MyRobot
from ..target import Target
from .dummy_movement import DummyMovement
from .robot_movement import RobotMovement

import time
import math

class Autonomous_RobotMovement(RobotMovement):

    def __init__(
        self,
        robot: MyRobot,
        target: Target,
        pr: PyRep,
        data_transforms,
        max_deviation: float=0.04,
        camera: VisionSensor =None,
        res: int=64
    ) -> None:

        super().__init__(robot, target, pr, max_deviation, camera, res)
        self.in_transform = data_transforms[0]
        self.eeVel_unnorm = data_transforms[1]
        self.recon_unnorm = data_transforms[2]

    def retransform_eeVel(self, v):
        if self.eeVel_unnorm is None:
            return v
        else:
            return self.eeVel_unnorm(v)


    def autonomousMovement(self, model: nn.Module):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        # img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        img: torch.Tensor = self.in_transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        # print(model(img).shape)
        v: torch.Tensor = model(img)[0]
        v = self.retransform_eeVel(v).detach().numpy()
        q = self.bot.get_jointVelo(v)
        self.robot.set_joint_target_velocities(q)
        self.pr.step()

    def autonomousMovement_constrained(self, model: nn.Module):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        # img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        img: torch.Tensor = self.in_transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        v: torch.Tensor = model(img)[0]
        v = self.retransform_eeVel(v).detach().numpy()
        q = self.bot.get_jointVelo_constrained(v[:3], v[3:])
        self.robot.set_joint_target_velocities(q)
        self.pr.step()

    def autonomousStop(self, model: nn.Module, constrained: bool=False):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        img: torch.Tensor = self.in_transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        # print(model(img).shape)
        v, stop = model(img)
        # print(out.shape)
        v = self.retransform_eeVel(v[0]).detach().numpy()
        stop = stop[0].detach().numpy()
        # print(v)
        if constrained:
            q = self.bot.get_jointVelo_constrained(v[:3], v[3:])
        else:
            q = self.bot.get_jointVelo(v)
        self.robot.set_joint_target_velocities(q)
        self.pr.step()
        # print(stop)
        return stop