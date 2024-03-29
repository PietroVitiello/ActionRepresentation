from abc import abstractmethod
from typing import List, Tuple
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape, JointMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape, Dummy
from pyrep.backend import sim

from ..target import Target
from .mico_gripper import MicoGripperComplete
from ..Kinematics.quadratic import Quadratic

import numpy as np
import math
import time as t

class MyRobot():

    def __init__(self, robot: Arm, hand: Gripper) -> None:
        self.robot = robot
        self.gripper = hand

        self.initialConf = None

        self.disable_controlLoop()
        self.fixJoints()

        self.robot_state = self.robot.get_configuration_tree()
        self.gripper_state = self.gripper.get_configuration_tree()
        robot_conf, gripper_conf = self.get_DefaultConfiguration()
        self.set_initialConf(robot_conf, gripper_conf)
        # print(f"Aperture: {self.gripper.get_joint_positions()}")

    @abstractmethod
    def get_DefaultConfiguration(self) -> Tuple[List[float], List[float]]:
        pass

    def disable_controlLoop(self):
        [j.set_control_loop_enabled(False) for j in self.robot.joints]
        [j.set_control_loop_enabled(False) for j in self.gripper.joints]

    def fixJoints(self) -> None:
        self.robot.set_joint_target_velocities([0]*(len(self.robot.joints)))
        self.gripper.set_joint_target_velocities([0]*(len(self.gripper.joints)))
        self.robot.set_motor_locked_at_zero_velocity(True)
        self.gripper.set_motor_locked_at_zero_velocity(True)

    def set_initialConf(self, robot_config: List[float], gripper_config: List[float]):
        self.initialConf = (robot_config, gripper_config)
        self.robot.set_joint_positions(robot_config)
        self.gripper.set_joint_positions(gripper_config)

    def resetInitial(self, pr: PyRep):
        self.robot.set_joint_target_velocities([0]*(len(self.robot.joints)))
        self.gripper.set_joint_target_velocities([0]*(len(self.gripper.joints)))
        self.robot.reset_dynamic_object()
        self.gripper.reset_dynamic_object()
        pr.set_configuration_tree(self.robot_state)
        pr.set_configuration_tree(self.gripper_state)
        
        self.robot.set_joint_positions(self.initialConf[0], disable_dynamics=True)
        self.gripper.set_joint_positions(self.initialConf[1], disable_dynamics=True)

    def getTip(self) -> Dummy:
        return self.robot._ik_tip

    def getRobot(self) -> Arm:
        return self.robot

    def getGripper(self) -> Gripper:
        return self.gripper

    @abstractmethod
    def close_gripper(self) -> bool:
        pass

    def get_trueJacobian(self):
        self.robot._ik_target.set_matrix(self.robot._ik_tip.get_matrix())
        sim.simCheckIkGroup(self.robot._ik_group,
                            [j.get_handle() for j in self.robot.joints])
        jacobian, (rows, cols) = sim.simGetIkGroupMatrix(self.robot._ik_group, 0)
        jacobian = np.array(jacobian).reshape((rows, cols), order='F')
        return np.flip(jacobian, axis=0)

    def findQuadratic(self, target: Target) -> Quadratic:
        return Quadratic(self.robot._ik_tip, target)

    def get_movementDir(self, target: Shape) -> np.ndarray:
        tip = self.robot._ik_tip.get_position()
        target = target.get_position()
        distance_vec = target - tip
        return distance_vec

    def get_linearVelo(self, direction: np.ndarray, time: float) -> np.ndarray:
        v = direction / time
        # R = self.robot.get_matrix()[:3,:3]
        return v #np.matmul(np.linalg.inv(R), v)

    def get_angularSpeed(self, target_or: np.ndarray, time: float=1) -> np.ndarray:
        # ee = self.robot.joints[-1]
        ee = self.robot._ik_tip
        # current_or = ee.get_orientation()
        # current_or = self.gripper.get_orientation()
        return target_or / time
        # return target_or / 0.05
        # return (target_or - current_or) / 5 #to get it to move in one time step

    def get_jointVelo(self, v: np.ndarray) -> np.ndarray:
        self.robot.set_ik_element_properties(constraint_alpha_beta=False, constraint_gamma=False)
        R = self.robot.get_matrix()[:3,:3]
        v = np.matmul(np.linalg.inv(R), v)
        J = self.get_trueJacobian()
        q = np.matmul(np.linalg.pinv(J.T), v)
        return q

    def get_jointVelo_constrained(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        # print(f"\n\n\nVelocity Jacobian: \n{self.robot.get_jacobian()}")
        self.robot.set_ik_element_properties()
        R = self.robot.get_matrix()[:3,:3]
        v = np.matmul(np.linalg.inv(R), v)
        vw = np.hstack((v, w))

        J = self.get_trueJacobian()
        # J = self.robot.get_jacobian()
        # print(f"\nJacobian: \n{J}")
        q = np.matmul(np.linalg.pinv(J.T), vw)
        return q

    def get_jointVelo_4orientation(self, w: np.ndarray) -> np.ndarray:
        self.robot.set_ik_element_properties(constraint_x=False, constraint_y=False, constraint_z=False)
        J = self.get_trueJacobian()
        q = np.matmul(np.linalg.pinv(J.T), w)
        return q

    def get_quaternion(self, target):
        ee = self.robot._ik_tip #self.robot.joints[-1]
        q = ee.get_quaternion(relative_to=target)
        return q

    def get_rotationAxis(self, q):
        theta = np.arccos(q[0])*2
        print("theta: ", theta, " which is ", math.degrees(theta))
        if theta > math.radians(10):
            axis = q[1:]/np.sin(theta/2)
            w = axis * theta / 5
            return w
        else:
            print("inside")
            return False

    # def stayStill(self, pr: PyRep, time: float):
    #     self.robot.set_joint_target_velocities([0]*7)
    #     n_steps = np.round(time / 0.05).astype(int)
    #     for _ in range(n_steps):
    #         pr.step()

    # def move_inDir(self, pr: PyRep, direction: np.ndarray, time: float):
    #     v = self.get_linearVelo(direction, time)
    #     n_steps = np.round(time / 0.05).astype(int)

    #     for i in range(n_steps):
    #         v = self.get_linearVelo(direction, time)
    #         q = self.get_jointVelo(v)
    #         self.robot.set_joint_target_velocities(q)
    #         pr.step()

    # def orientArm(self, pr: PyRep, orientation: np.ndarray, time: float):
    #     v = np.array([0, 0, 0])
    #     n_steps = np.round(time / 0.05).astype(int)

    #     dummy = Dummy.create()
    #     dummy.set_orientation(orientation)

    #     for i in range(n_steps):
    #         orientation = dummy.get_orientation(relative_to=self.robot._ik_tip)
    #         w = self.get_angularSpeed(orientation)
    #         q = self.get_jointVelo_constrained(v, w)
    #         self.robot.set_joint_target_velocities(q)
    #         pr.step()
    #         time -= 0.05

    # def moveArm(self, pr:PyRep, target: Shape, time: float=2):
    #     n_steps = np.round(time / 0.05).astype(int)
    #     # print(n_steps)

    #     for i in range(n_steps):
    #         distance = self.get_movementDir(target)
    #         v = self.get_linearVelo(distance, time)
    #         # print(f"\nVelocity magnitude: {np.linalg.norm(v)}")
    #         # print(f"Direction: {distance}")
    #         # print(f"Target: {target.get_position()}")
    #         # print(f"Position: {self.robot._ik_tip.get_position()}")
    #         q = self.get_jointVelo(v)
    #         self.robot.set_joint_target_velocities(q)
    #         t.sleep(0.5)
    #         pr.step()
    #         time -= 0.05

    # def moveArm_constrained(self, pr:PyRep, target: Shape, time: float=2):
    #     curve = self.findQuadratic(target)
    #     n_steps = np.round(time / 0.05).astype(int)

    #     for i in range(n_steps):
    #         distance = self.get_movementDir(target)
    #         orientation = curve.linear_mid.get_orientation(relative_to=self.robot._ik_tip)
    #         v = self.get_linearVelo(distance, time)
    #         w = self.get_angularSpeed(orientation)
    #         q = self.get_jointVelo_constrained(v, w)
    #         self.robot.set_joint_target_velocities(q)
    #         # t.sleep(0.5)
    #         pr.step()
    #         time -= 0.05
    #     curve.remove_dummies()

    # def moveArmCurved(self, pr:PyRep, target: Shape, time: float=2): #(self, pos: np.ndarray, target: np.ndarray, n_steps: float):
    #     curve = self.findQuadratic(target)
    #     curve.find_middlePoint()
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance = self.get_movementDir(target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (curve.get_arcLen()/time) * direction

    #     for _ in range(n_steps-8):
    #         v = curve.get_tangentVelocity(v_lin)
    #         q = self.get_jointVelo(v)
    #         self.robot.set_joint_target_velocities(q)
    #         pr.step()
    #     curve.remove_dummies()

    # def moveArmCurved_constrained(self, pr:PyRep, target: Shape, time: float=2):
    #     curve = self.findQuadratic(target)
    #     curve.find_middlePoint()
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance = self.get_movementDir(target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (curve.get_arcLen()/time) * direction

    #     print("cavolfiore: ", 13/n_steps)

    #     for _ in range(n_steps-13):
    #         orientation = curve.get_FaceTargetOrientation()
    #         v = curve.get_tangentVelocity(v_lin)
    #         w = self.get_angularSpeed(orientation)
    #         q = self.get_jointVelo_constrained(v, w)
    #         self.robot.set_joint_target_velocities(q)
    #         pr.step()
    #     curve.remove_dummies()

    
    # def nana(self, pr: PyRep, target):
    #     curve = Quadratic(self.robot._ik_tip.get_position(), target.get_position())
    #     v = np.array([0, 0, 0])
    #     for _ in range(600):
    #         q = self.get_quaternion(curve.linear_mid)
    #         w = self.get_rotationAxis(q)
    #         if type(w) == np.ndarray:
    #             print("doing")
    #             q = self.get_jointVelo_constrained(v, w)
    #             self.robot.set_joint_target_velocities(q)
    #         else:
    #             self.robot.set_joint_target_velocities([0]*7)
    #         pr.step()
        



        


    