import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape, Dummy
from quadratic import Quadratic

from my_robot import MyRobot
from target import Target
from dummy_movement import DummyMovement

import time
import math

class RobotMovement():

    def __init__(self, robot: MyRobot, target: Target, pr: PyRep) -> None:
        self.bot = robot
        self.robot = self.bot.getRobot()
        self.target = target
        self.pr = pr

        self.curve = Quadratic(self.bot.getTip(), self.target, 0.08)

    def resetCurve(self) -> None:
        self.curve.resetCurve()

    def stayStill(self, time: float):
        self.robot.set_joint_target_velocities([0]*7)
        n_steps = np.round(time / 0.05).astype(int)
        for _ in range(n_steps):
            # print("debug: ", self.bot.gripper.get_joint_positions())
            self.pr.step()

    def move_inDir(self, direction: np.ndarray, time: float):
        v = self.bot.get_linearVelo(direction, time)
        n_steps = np.round(time / 0.05).astype(int)

        for i in range(n_steps):
            v = self.bot.get_linearVelo(direction, time)
            q = self.bot.get_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()

    def orientArm(self, orientation: np.ndarray, time: float):
        v = np.array([0, 0, 0])
        n_steps = np.round(time / 0.05).astype(int)

        dummy = Dummy.create()
        dummy.set_orientation(orientation)

        for i in range(n_steps):
            orientation = dummy.get_orientation(relative_to=self.robot._ik_tip)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            time -= 0.05

    def moveArm(self, time: float=2):
        n_steps = np.round(time / 0.05).astype(int)
        # print(n_steps)

        for i in range(n_steps):
            distance = self.bot.get_movementDir(self.target)
            v = self.bot.get_linearVelo(distance, time)
            # print(f"\nVelocity magnitude: {np.linalg.norm(v)}")
            # print(f"Direction: {distance}")
            # print(f"Target: {target.get_position()}")
            # print(f"Position: {self.robot._ik_tip.get_position()}")
            q = self.bot.get_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            # t.sleep(0.5)
            self.pr.step()
            time -= 0.05

    def moveArm_constrained(self, time: float=2):
        n_steps = np.round(time / 0.05).astype(int)

        for i in range(n_steps):
            distance = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.robot._ik_tip)
            v = self.bot.get_linearVelo(distance, time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            # t.sleep(0.5)
            self.pr.step()
            time -= 0.05
        self.curve.remove_dummies()

    def moveArmCurved(self, time: float=2):
        self.curve.find_middlePoint()
        n_steps = np.round(time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/time) * direction

        for _ in range(n_steps-8):
            v = self.curve.get_tangentVelocity(v_lin)
            q = self.bot.get_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
        self.curve.remove_dummies()

    def moveArmCurved_constrained(self, time: float=2):
        self.curve.find_middlePoint()
        n_steps = np.round(time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/time) * direction

        print("cavolfiore: ", 13/n_steps)

        for _ in range(n_steps-13):
            orientation = self.curve.get_FaceTargetOrientation()
            v = self.curve.get_tangentVelocity(v_lin)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
        self.curve.remove_dummies()

    def humanMovement(self, time: float):
        self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, time)
        n_steps = np.round(time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/time) * direction

        # print("cavolfiore: ", 13/n_steps)

        for i in range(n_steps):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            # v = self.curve.get_tangentVelocity(v_lin)
            v = self.curve.get_enhancedTangentVelocity(v_lin, time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            # q = self.bot.get_jointVelo(v)
            if self.check_cubeReached():
                i = n_steps + 1
                print("Cube Reached")
                q = np.zeros(q.shape)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            dmove.step()
            print("step: ", i+1)
        self.curve.remove_dummies()
        dmove.remove_dummy()

    def check_cubeReached(self, threshold=0.04) -> bool:
        distance = self.target.get_position() - self.robot._ik_tip.get_position()
        distance = np.linalg.norm(distance)
        return True if distance <= threshold else False





# pr = PyRep()
# plt.ion()

# SCENE_FILE = join(dirname(abspath(__file__)), "Simulations/coppelia_robot_arm_copy.ttt")
# pr.launch(SCENE_FILE, headless=False)
# pr.start()
# pr.step_ui()

# # arm = get_arm()
# bot = MyRobot()
# target = Target()
# camera = VisionSensor("Vision_sensor")

# # target.set_position([0, 1, 0.1])

# # bot.find_jointVelo(target, 40)

# # target.random_pos()
# # bot.trajetoryNoise(target)
# # bot.stayStill(pr, 100)

# for _ in range(5):
#     target.random_pos()
#     bot.resetInitial()
#     bot.stayStill(pr, 1)
#     bot.moveArm_constrained(pr, target, 4)

# # for _ in range(5):
# #     target.random_pos()
# #     # target.set_position([0.45, 0.55, 0.025])
# #     bot.resetInitial()
# #     bot.stayStill(pr, 1)
# #     bot.moveArmCurved(pr, target, 3)

# # for _ in range(10):
# #     target.random_pos()
# #     bot.resetInitial()
# #     bot.stayStill(pr, 1)
# #     bot.trajetoryNoise(pr, target, 5)

# # initialConf = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
# # bot.robot.set_joint_positions(initialConf)
# # bot.stayStill(pr, 1000)

# # i=0
# # while i<10e6:
# #     print(i)
# #     i+=1

# # bot.move_inDir(pr, np.array([0, 0, 1]), 20)

# # bot.stayStill(pr, 1)
# # bot.nana(pr, target)

# pr.stop()
# pr.shutdown()

