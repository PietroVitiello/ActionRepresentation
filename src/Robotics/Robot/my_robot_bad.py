import numpy as np
import math
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

        # self.robot.set_joint_mode(JointMode.IK)

    def disable_controlLoop(self):
        [j.set_control_loop_enabled(False) for j in self.robot.joints]

    def set_initialConf(self, config):
        self.initialConf = config
        self.robot.set_joint_positions(config)

    def resetInitial(self):
        self.robot.set_joint_positions(self.initialConf)

    def get_EErotation(self):
        # ee = self.robot.joints[6]
        ee = self.robot.joints[-1]
        # ee = self.robot._ik_tip
        # print(ee.get_matrix())
        return ee.get_matrix()

    def get_movementDir(self, target: Shape) -> np.ndarray:
        tip = self.robot._ik_tip.get_position()
        target = target.get_position()
        # self.robot._ik_target.set_position(target)
        direction = target - tip
        return direction

    def get_linearVelo(self, direction: np.ndarray, time: float) -> np.ndarray:
        v = direction / 20 #time
        # print("v ", v)
        R = self.robot.get_matrix()[:3,:3]
        # R = self.get_EErotation()[:3,:3]
        # print("\noriginal ", v)
        # print(np.matmul(np.linalg.inv(R), v))
        # return v
        return np.matmul(np.linalg.inv(R), v)

    def find_jointVelo(self, v: np.ndarray) -> np.ndarray:
        # J = self.robot.get_jacobian()
        J = self.crazy_Jacobian()
        q = np.matmul(np.linalg.pinv(J.T), v)
        # print(q)
        return q

    def move_inDir(self, pr: PyRep, direction: np.ndarray, time: float):
        v = self.get_linearVelo(direction, time)
        n_steps = np.round(time / 0.05).astype(int)

        for i in range(n_steps):
            v = self.get_linearVelo(direction, time)
            q = self.find_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            pr.step()

    def move_arm(self, pr:PyRep, target: Shape, n_steps: int, time: float=2):
        direction = self.get_movementDir(target)
        v = self.get_linearVelo(direction, time)

        for i in range(n_steps):
            direction = self.get_movementDir(target)
            # print(direction)
            v = self.get_linearVelo(direction, time)
            print(f"\nVelocity magnitude: {np.linalg.norm(v)}")
            print(f"Direction: {direction}")
            print(f"Target: {target.get_position()}")
            print(f"Position: {self.robot._ik_tip.get_position()}")
            # print("rotated v ", v)
            q = self.find_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            pr.step()

    def crazy_Jacobian(self):
        # print("terrible: \n", self.terrible())
        J = self.robot.get_jacobian()
        J_new3 = J
        # print(J)
        # print("\n\n1: \n")
        # print(np.matmul(J.T, np.array([1,0,0,0,0,0,0])))
        # print("\n\n2: \n")
        # print(np.matmul(J.T, np.array([0,1,0,0,0,0,0])))
        # print("\n\n3: \n")
        # print(np.matmul(J.T, np.array([0,0,1,0,0,0,0])))
        # print("\n\n4: \n")
        # print(np.matmul(J.T, np.array([0,0,0,1,0,0,0])))
        # print("\n\n5: \n")
        # print(np.matmul(J.T, np.array([0,0,0,0,1,0,0])))
        # print("\n\n6: \n")
        # print(np.matmul(J.T, np.array([0,0,0,0,0,1,0])))
        # print("\n\n7: \n")
        # print(np.matmul(J.T, np.array([0,0,0,0,0,0,1])))
        row, col = J.shape
        J = np.ravel(J, order='F')
        J_new = np.zeros((row, col))
        J_new2 = np.zeros((col, row))
        J_new4 = J_new
        for i in range(row):
            for j in range(col):
                J_new[i][col-j-1] = J[i*col + j]

        for i in range(row):
            for j in range(col):
                J_new2[j][col-i-1] = J[j*col + i]

        for i in range(row):
            J_new4[row-i-1,:] = J_new3[i]

        # print(J_new4)
        # print("\n", J_new3)
        # print("\nJ mine: \n", J_new)
        # print("\n\nJ his: \n", J_new2)

        # J_new = np.reshape(J, (row, col))
        return J_new4

    def terrible(self):
        self.robot._ik_target.set_matrix(self.robot._ik_tip.get_matrix())
        sim.simCheckIkGroup(self.robot._ik_group,
                            [j.get_handle() for j in self.robot.joints])
        jacobian, (rows, cols) = sim.simGetIkGroupMatrix(self.robot._ik_group, 0)
        # jacobian = np.array(jacobian).reshape((rows, cols), order='F')
        return jacobian
            
    def nana(self):
        [print(j.get_joint_velocity())  # type: ignore
         for j in self.robot.joints]
        print("\n\n")

        [print(j.get_joint_target_velocity()) # type: ignore
         for j in self.robot.joints]
        print("\n\n\n")

        [print(j.is_control_loop_enabled()) # type: ignore
         for j in self.robot.joints]
        print("\n\n\n")

        [print(j.get_joint_mode()) # type: ignore
         for j in self.robot.joints]



    