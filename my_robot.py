import matplotlib.pyplot as plt
import numpy as np
import math
# import time as t
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape, JointMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

from pyrep.backend import sim

class MyRobot():

    def __init__(self) -> None:
        self.robot = LBRIwaa14R820()
        self.gripper = MicoGripper()

        self.initialConf = None

        self.disable_controlLoop()

        # initialConf = [0, math.radians(-30), 0, math.radians(-50), 0, math.radians(20), 0]
        # initialConf = [0, 0, 0, math.radians(-90), 0, math.radians(0), 0]
        initialConf = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
        self.set_initialConf(initialConf)
        self.robot.set_motor_locked_at_zero_velocity(True)

    def disable_controlLoop(self):
        [j.set_control_loop_enabled(False) for j in self.robot.joints]

    def set_initialConf(self, config):
        self.initialConf = config
        self.robot.set_joint_positions(config)

    def resetInitial(self):
        self.robot.set_joint_positions(self.initialConf, disable_dynamics=True)

    def stayStill(self, pr: PyRep, time: float):
        self.robot.set_joint_target_velocities([0]*7)
        n_steps = np.round(time / 0.05).astype(int)
        for _ in range(n_steps):
            pr.step()

    def get_trueJacobian(self):
        self.robot._ik_target.set_matrix(self.robot._ik_tip.get_matrix())
        sim.simCheckIkGroup(self.robot._ik_group,
                            [j.get_handle() for j in self.robot.joints])
        jacobian, (rows, cols) = sim.simGetIkGroupMatrix(self.robot._ik_group, 0)
        jacobian = np.array(jacobian).reshape((rows, cols), order='F')
        return np.flip(jacobian, axis=0)

    def get_movementDir(self, target: Shape) -> np.ndarray:
        tip = self.robot._ik_tip.get_position()
        target = target.get_position()
        distance = target - tip
        return distance

    def get_linearVelo(self, direction: np.ndarray, time: float) -> np.ndarray:
        v = direction / time
        R = self.robot.get_matrix()[:3,:3]
        return np.matmul(np.linalg.inv(R), v)

    def find_jointVelo(self, v: np.ndarray) -> np.ndarray:
        J = self.get_trueJacobian()
        q = np.matmul(np.linalg.pinv(J.T), v)
        return q

    def move_inDir(self, pr: PyRep, direction: np.ndarray, time: float):
        v = self.get_linearVelo(direction, time)
        n_steps = np.round(time / 0.05).astype(int)

        for i in range(n_steps):
            v = self.get_linearVelo(direction, time)
            q = self.find_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            pr.step()

    def move_arm(self, pr:PyRep, target: Shape, time: float=2):
        n_steps = np.round(time / 0.05).astype(int)
        # print(n_steps)

        for i in range(n_steps):
            distance = self.get_movementDir(target)
            v = self.get_linearVelo(distance, time)
            # print(f"\nVelocity magnitude: {np.linalg.norm(v)}")
            # print(f"Direction: {distance}")
            # print(f"Target: {target.get_position()}")
            # print(f"Position: {self.robot._ik_tip.get_position()}")
            q = self.find_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            # t.sleep(0.04)
            pr.step()
            time -= 0.05

    def trajetoryNoise(self, pos: np.ndarray, target: np.ndarray, n_steps: float):
        direction = target - pos
        theta = np.random.uniform(-180, 180)
        


    