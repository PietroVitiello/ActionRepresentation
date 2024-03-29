import numpy as np
from os.path import dirname, join, abspath

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

import time
import math

class DummyMovement():

    def __init__(self, target: Target, time: float, tip: Dummy=None, theta: float=None) -> None:
        self.target = target
        self.time = time * 0.8 #0.9
        self.dummy = None
        if tip is not None: 
            self.generateDummy_inFront(tip)
            self.time = 0.6 * time
        elif theta is None:
            self.random_generateDummy()
        else:
            self.generateDummy_alongDir(theta + np.pi/2)

    def generateDummy_alongDir(self, theta):
        # distance = np.random.uniform(0.1, 0.15)
        x = np.cos(theta)
        y = np.sin(theta)
        distance = 0.15

        rel_pos = np.array([x, y, 0])*distance
        self.dummy = Dummy.create(0.05)#(0.001)
        self.dummy.set_position(self.target.get_position() + rel_pos)

    def random_generateDummy(self):
        x = np.random.uniform(-1, 1)
        y = np.sqrt(1 - x**2)
        y_sign = 1 if np.random.uniform() < 0.5 else -1
        distance = np.random.uniform(0.1, 0.15)
        # distance = 0.7
        rel_pos = np.array([x, y*y_sign, 0])*distance
        self.dummy = Dummy.create(0.001)
        self.dummy.set_position(self.target.get_position() + rel_pos)

    def generateDummy_inFront(self, tip: Dummy):
        x_dir = tip.get_matrix()[:,0]
        target_elevation = self.target.get_position()[2]
        tip_pos = tip.get_position()
        vectors_needed = (target_elevation - tip_pos[2]) / x_dir[2]
        displacement = x_dir[:2] * vectors_needed
        projected_pos = tip_pos[:2] + displacement
        self.dummy = Dummy.create(0.001)
        self.dummy.set_position([*projected_pos, target_elevation])

    def remove_dummy(self):
        self.dummy.remove()

    def getDummy(self) -> Dummy:
        return self.dummy

    def get_movementDir(self) -> np.ndarray:
        tip = self.dummy.get_position()
        target = self.target.get_position()
        distance = target - tip
        return distance

    def getVelocity(self):
        distance = self.get_movementDir()
        return distance / self.time

    def step(self):
        pos = self.dummy.get_position()
        v = self.getVelocity() * 0.05
        self.dummy.set_position(pos + v)
        self.time -= 0.05

