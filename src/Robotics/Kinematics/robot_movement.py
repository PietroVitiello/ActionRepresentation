import matplotlib.pyplot as plt
import numpy as np
import torch
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

import time
import math

class RobotMovement():

    def __init__(self, robot: MyRobot, target: Target, pr: PyRep, camera: VisionSensor =None, res: int=64) -> None:
        self.bot = robot
        self.robot = self.bot.getRobot()
        self.target = target
        self.pr = pr

        self.curve = Quadratic(self.bot.getTip(), self.target, 0.08) #0.0001
        if camera is not None:
            self.camera = camera
            self.camera.set_resolution([res, res])

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

    def imperfect_humanMovement(self, time: float):
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

    def humanMovement(self, time: float):
        #only velocity
        theta = self.curve.find_middlePoint()
        n_steps = np.round(time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/time) * direction

        for i in range(n_steps):
            v = self.curve.getVelocity2Target(v_lin)
            q = self.bot.get_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            if self.check_cubeReached():
                print(f"Cube Reached at step {i+1}")
                break
        self.curve.remove_dummies()

    # def humanMovement(self, time: float):
    #     theta = self.curve.find_middlePoint()
    #     dmove = DummyMovement(self.target, time)
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance = self.bot.get_movementDir(self.target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (self.curve.get_arcLen()/time) * direction

    #     # print("cavolfiore: ", 13/n_steps)

    #     for i in range(n_steps):
    #         # orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
    #         orientation = self.curve.get_FaceTargetOrientation()
    #         # v = self.curve.get_tangentVelocity(v_lin)
    #         # v = self.curve.get_enhancedTangentVelocity(v_lin, time)
    #         v = self.curve.getVelocity2Target(v_lin)
    #         w = self.bot.get_angularSpeed(orientation)
    #         # w = np.array([0,0,0])
    #         q = self.bot.get_jointVelo_constrained(v, w)
    #         # q = self.bot.get_jointVelo(v)
    #         if self.check_cubeReached():
    #             print(f"Cube Reached at step {i+1}")
    #             q = np.zeros(q.shape)
    #             self.robot.set_joint_target_velocities(q)
    #             break
    #         self.robot.set_joint_target_velocities(q)
    #         self.pr.step()
    #         dmove.step()
    #         # print("step: ", i)
    #     self.curve.remove_dummies()
    #     dmove.remove_dummy()

    # def humanMovement(self, time: float):
    #     theta = self.curve.find_middlePoint()
    #     dmove = DummyMovement(self.target, time)
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance = self.bot.get_movementDir(self.target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (self.curve.get_arcLen()/time) * direction

    #     # print("cavolfiore: ", 13/n_steps)

    #     for i in range(n_steps):
    #         # orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
    #         orientation = self.curve.get_FaceTargetOrientation()
    #         # v, verga = self.curve.getVelocity2Target(v_lin)
    #         # orientation = self.curve.get_FaceTargetOrientation(verga)
    #         v_standard = self.curve.get_tangentVelocity(v_lin)
    #         # v_standard = self.curve.get_enhancedTangentVelocity(v_lin, time)
    #         v, _ = self.curve.getVelocity2Target(v_lin, i)
    #         w = self.bot.get_angularSpeed(orientation)
    #         # w /= 0.05
    #         # w = np.array([0,0,0])
    #         # v = [0,0,0]

    #         # print("without w: ", self.bot.get_jointVelo(v))
    #         # print("with w=0: ", self.bot.get_jointVelo_constrained(v, w), "\n")

    #         print("vw: ", np.hstack((v, w)))
    #         print("vw standard: ", np.hstack((v_standard, w)))
    #         print("vw standard: ", v - v_standard, "\n")

    #         q = self.bot.get_jointVelo_constrained(v, w)
    #         # q = self.bot.get_jointVelo(v)
    #         # q = self.bot.get_jointVelo_4orientation(w)
    #         q_standard = self.bot.get_jointVelo_constrained(v_standard, w)

    #         print("q: ", q)
    #         print("q standard: ", q_standard, "\n\n")

    #         if self.check_cubeReached():
    #             print(f"Cube Reached at step {i+1}")
    #             q = np.zeros(q.shape)
    #             self.robot.set_joint_target_velocities(q)
    #             break
    #         self.robot.set_joint_target_velocities(q)
    #         # input("Next step")
    #         self.pr.step()
    #         dmove.step()
    #         # print("step: ", i)
    #     self.curve.remove_dummies()
    #     dmove.remove_dummy()

    # def humanMovement(self, time: float):
    #     [j.set_control_loop_enabled(True) for j in self.robot.joints]
    #     theta = self.curve.find_middlePoint()
    #     dmove = DummyMovement(self.target, time)
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance = self.bot.get_movementDir(self.target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (self.curve.get_arcLen()/time) * direction

    #     # print("cavolfiore: ", 13/n_steps)

    #     for i in range(n_steps):
    #         target_pos = self.curve.getVelocity2Target(v_lin, i)
    #         config = self.robot.get_configs_for_tip_pose(position=target_pos, euler=dmove.getDummy().get_orientation())
    #         print("configs:\n", config)
    #         self.robot.set_joint_target_positions(config[0])
            
    #         # input("Next step")
    #         self.pr.step()
    #         dmove.step()
    #         # print("step: ", i)
    #     self.curve.remove_dummies()
    #     dmove.remove_dummy()

    def check_cubeReached(self, threshold=0.04) -> bool:
        distance = self.target.get_position() - self.robot._ik_tip.get_position()
        distance = np.linalg.norm(distance)
        # print("distance: ", distance)
        # print(self.robot._ik_tip.get_position())
        return True if distance <= threshold else False

    def autonomousMovement(self, model, transform):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        # img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        img: torch.Tensor = transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        # print(model(img).shape)
        v: torch.Tensor = model(img) #[0]
        v = v.detach().numpy()
        q = self.bot.get_jointVelo(v)
        self.robot.set_joint_target_velocities(q)
        self.pr.step()

    def autonomousMovement_constrained(self, model, transform):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        # img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        img: torch.Tensor = transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        v: torch.Tensor = model(img)[0]
        v = v.detach().numpy()
        q = self.bot.get_jointVelo_constrained(v[:3], v[3:])
        self.robot.set_joint_target_velocities(q)
        self.pr.step()

    def graspingMovement(self, time: float):

        # grasped = False
        # while not grasped:
        #     grasped = self.bot.close_gripper()
        #     self.pr.step()
        #     print(grasped)
        # print("Cube Grasped")

        while True:
            # print(self.bot.gripper.get_joint_target_velocities())
            self.pr.step()

        # while not self.check_cubeReached(0.03):
        #     distance = self.bot.get_movementDir(self.target)
        #     orientation = self.curve.linear_mid.get_orientation(relative_to=self.robot._ik_tip)
        #     v = self.bot.get_linearVelo(distance, time)
        #     w = self.bot.get_angularSpeed(orientation)
        #     q = self.bot.get_jointVelo_constrained(v, w)
        #     self.robot.set_joint_target_velocities(q)
        #     self.pr.step()
        #     time -= 0.05

        # print("Cube Reached")
        
        # q = [0]*len(q)
        # self.robot.set_joint_target_velocities(q)
        # grasped = False
        # while not grasped:
        #     grasped = self.bot.close_gripper()
        #     self.pr.step()
        #     print(grasped)
        # print("Cube Grasped")
        self.curve.remove_dummies()





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

