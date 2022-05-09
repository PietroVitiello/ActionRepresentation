from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape, JointMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.backend import sim

from quadratic import Quadratic

import matplotlib.pyplot as plt
import numpy as np
import math

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
        # R = self.robot.get_matrix()[:3,:3]
        return v #np.matmul(np.linalg.inv(R), v)

    def find_jointVelo(self, v: np.ndarray) -> np.ndarray:
        R = self.robot.get_matrix()[:3,:3]
        v = np.matmul(np.linalg.inv(R), v)
        J = self.get_trueJacobian()
        q = np.matmul(np.linalg.pinv(J.T), v)
        return q

    def get_angularSpeed(self, target_or: np.ndarray) -> np.ndarray:
        ee = self.robot.joints[-1]
        current_or = ee.get_orientation()
        # current_or = self.gripper.get_orientation()
        return (current_or - target_or) / 5 #to get it to move in one time step

    def get_jointVelo_constrained(self, v: np.ndarray, w: np.ndarray):
        # print(f"\n\n\nVelocity Jacobian: \n{self.robot.get_jacobian()}")
        self.robot.set_ik_element_properties()
        R = self.robot.get_matrix()[:3,:3]
        v = np.matmul(np.linalg.inv(R), v)
        vw = np.hstack((v, w))
        print(vw)

        J = self.get_trueJacobian()
        # J = self.robot.get_jacobian()
        # print(f"\nJacobian: \n{J}")
        q = np.matmul(np.linalg.pinv(J.T), vw)
        return q

    def move_inDir(self, pr: PyRep, direction: np.ndarray, time: float):
        v = self.get_linearVelo(direction, time)
        n_steps = np.round(time / 0.05).astype(int)

        for i in range(n_steps):
            v = self.get_linearVelo(direction, time)
            q = self.find_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            pr.step()

    def moveArm(self, pr:PyRep, target: Shape, time: float=2):
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

    def stayStill(self, pr: PyRep, time: float):
        self.robot.set_joint_target_velocities([0]*7)
        n_steps = np.round(time / 0.05).astype(int)
        for _ in range(n_steps):
            pr.step()

    def trajetoryNoise(self, pr:PyRep, target: Shape, time: float=2): #(self, pos: np.ndarray, target: np.ndarray, n_steps: float):
        curve = Quadratic(self.robot._ik_tip.get_position(), target.get_position())
        curve.find_middlePoint()
        n_steps = np.round(time / 0.05).astype(int)

        distance = self.get_movementDir(target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (curve.get_arcLen()/time) * direction

        for _ in range(n_steps):
            v = curve.get_tangentVelocity(self.robot._ik_tip, v_lin)
            q = self.find_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            pr.step()
        
        curve.rem()

    def moveArm_constrained(self, pr: PyRep):
        v = np.array([0, 0, 0])
        # target_or = self.gripper.get_orientation() + np.array([math.radians(45),0, 0])
        ee = self.robot.joints[-1]
        target_or = ee.get_orientation() + np.array([0, 0, math.radians(45)])
        for _ in range(600):
            w = self.get_angularSpeed(target_or)
            # w = np.array([0,0.05,0])
            q = self.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            pr.step()



        


    