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

import time
import math

class RobotMovement():

    def __init__(self, robot: MyRobot, target: Target, pr: PyRep, max_deviation: float=0.04, camera: VisionSensor =None, res: int=64) -> None:
        self.bot = robot
        self.robot = self.bot.getRobot()
        self.target = target
        self.pr = pr

        self.curve = Quadratic(self.bot.getTip(), self.target, max_deviation) #0.08) #0.0001
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

    def orientArm_inPlace(self, orientation: np.ndarray, time: float):
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

    def orientArm(self, orientation: np.ndarray, time: float):
        n_steps = np.round(time / 0.05).astype(int)
        dummy = Dummy.create()
        dummy.set_orientation(orientation, relative_to=self.robot._ik_tip)

        for i in range(n_steps):
            orientation = dummy.get_orientation(relative_to=self.robot._ik_tip)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_4orientation(w)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            time -= 0.05

    def displaceArm(self, destination: np.ndarray, time: float=2):
        n_steps = np.round(time / 0.05).astype(int)
        for _ in range(n_steps):
            distance_vec = destination - self.robot._ik_tip.get_position()
            v = self.bot.get_linearVelo(distance_vec, time)
            q = self.bot.get_jointVelo(v)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            # print(self.bot.gripper.get_joint_target_velocities())
            time -= 0.05

    def moveArm(self, time: float=2):
        n_steps = np.round(time / 0.05).astype(int)
        # print(n_steps)

        for i in range(n_steps):
            distance_vec = self.bot.get_movementDir(self.target)
            v = self.bot.get_linearVelo(distance_vec, time)
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
            distance_vec = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.robot._ik_tip)
            v = self.bot.get_linearVelo(distance_vec, time)
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

        distance_vec = self.bot.get_movementDir(self.target)
        direction = distance_vec / np.linalg.norm(distance_vec)
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

        distance_vec = self.bot.get_movementDir(self.target)
        direction = distance_vec / np.linalg.norm(distance_vec)
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

    def demoMovement(self, time: float):
        self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, time)
        n_steps = np.round(time / 0.05).astype(int)

        distance_vec = self.bot.get_movementDir(self.target)
        direction = distance_vec / np.linalg.norm(distance_vec)
        v_lin = (self.curve.get_arcLen()/time) * direction

        n_steps_imp = int(np.floor(n_steps*0.80))
        n_steps = n_steps - n_steps_imp

        for i in range(n_steps_imp):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.get_tangentVelocity(v_lin)
            # v = self.curve.get_enhancedTangentVelocity(v_lin, time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            # q = self.bot.get_jointVelo(v)
            # if self.check_cubeReached():
            #     i = n_steps + 1
            #     print("Cube Reached")
            #     q = np.zeros(q.shape)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            dmove.step()

        time = time*0.15

        for i in range(n_steps):
            distance_vec = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.robot._ik_tip)
            v = self.bot.get_linearVelo(distance_vec, time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            # t.sleep(0.5)
            input()
            self.pr.step()
            time -= 0.05
            if self.check_cubeReached():
                print(f"Cube Reached at step {i+1}")
                break

        # for i in range(n_steps):
        #     orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
        #     v = self.curve.getVelocity2Target(v_lin)
        #     w = self.bot.get_angularSpeed(orientation, time=0.05)
        #     q = self.bot.get_jointVelo_constrained(v, w)
        #     self.robot.set_joint_target_velocities(q)
        #     self.pr.step()
        #     dmove.step()
            # if self.check_cubeReached():
            #     print(f"Cube Reached at step {i+1}")
            #     break

        self.curve.remove_dummies()
        dmove.remove_dummy()

    def imperfect_humanMovement(self, time: float):
        theta = self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, time, theta=theta)
        n_steps = np.round(time / 0.05).astype(int)

        distance_vec = self.bot.get_movementDir(self.target)
        direction = distance_vec / np.linalg.norm(distance_vec)
        v_lin = (self.curve.get_arcLen()/time) * direction

        for i in range(n_steps):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.get_tangentVelocity(v_lin)
            # v = self.curve.get_enhancedTangentVelocity(v_lin, time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            if self.check_cubeReached():
                i = n_steps + 1
                print("Cube Reached")
                q = np.zeros(q.shape)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            dmove.step()
        self.curve.remove_dummies()
        dmove.remove_dummy()

    def curvedMovement_followDummy(self, time: float):
        #only velocity
        _ = self.curve.find_middlePoint()
        n_steps = np.round(time / 0.05).astype(int)

        distance_vec = self.bot.get_movementDir(self.target)
        direction = distance_vec / np.linalg.norm(distance_vec)
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
        # _ = self.curve.find_middlePoint()
        # faceCube_orientation = self.curve.get_FaceTargetOrientation()
        # orientation_time = 0.20
        # self.orientArm(faceCube_orientation, time=orientation_time)
        # time -= orientation_time

        # _ = self.curve.find_middlePoint()
        # dmove = DummyMovement(self.target, time, tip=self.bot.getTip())
        # n_steps = np.round(time / 0.05).astype(int)

        # distance = self.bot.get_movementDir(self.target)
        # direction = distance / np.linalg.norm(distance)
        # v_lin = (self.curve.get_arcLen()/time) * direction

        # for i in range(n_steps):
        #     orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
        #     v = self.curve.getVelocity2Target(v_lin)
        #     w = self.bot.get_angularSpeed(orientation)
        #     q = self.bot.get_jointVelo_constrained(v, w)
        #     self.robot.set_joint_target_velocities(q)
        #     self.pr.step()
        #     dmove.step()
        #     if self.check_cubeReached():
        #         print(f"Cube Reached at step {i+1}")
        #         break
        # self.curve.remove_dummies()
        # dmove.remove_dummy()

    def humanMovement(self, time: float):
        _ = self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, time, tip=self.bot.getTip())
        n_steps = np.round(time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/time) * direction

        for i in range(n_steps):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.getVelocity2Target(v_lin)
            w = self.bot.get_angularSpeed(orientation, time=0.05)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            dmove.step()
            if self.check_cubeReached():
                print(f"Cube Reached at step {i+1}")
                break
        self.curve.remove_dummies()
        dmove.remove_dummy()

    def graspingMovement_linear(self, time: float):

        # grasped = False
        # print(self.bot.gripper.get_joint_positions())
        # while  not grasped:
        #     grasped = self.bot.close_gripper()
        #     self.pr.step()
        #     print(self.bot.gripper.get_joint_positions())
        #     print(grasped)
        # print("Cube Grasped")

        # while True:
        #     # print(self.bot.gripper.get_joint_target_velocities())
        #     print(self.bot.gripper.get_joint_positions())
        #     self.pr.step()

        while not self.check_cubeReached(0.01):
            distance = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.robot._ik_tip)
            v = self.bot.get_linearVelo(distance, time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            time -= 0.05

        print("Cube Reached")
        
        q = [0]*len(q)
        self.robot.set_joint_target_velocities(q)
        grasped = False
        while not grasped:
            grasped = self.bot.close_gripper()
            self.pr.step()
            # print(grasped)
            # print(self.bot.gripper.get_joint_forces())
        print("Cube Grasped")
        self.stayStill(0.1)
        # print(f"Last force: {self.bot.gripper.get_joint_forces()}")

        destination = self.bot.getTip().get_position()
        destination[2] += 0.3
        self.displaceArm(destination)

        self.curve.remove_dummies()

    def graspingMovement_human(self, time: float):

        _ = self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, time, tip=self.bot.getTip())

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/time) * direction

        while not self.check_cubeReached(0.025):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.getVelocity2Target(v_lin)
            w = self.bot.get_angularSpeed(orientation, time=0.05)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.robot.set_joint_target_velocities(q)
            self.pr.step()
            dmove.step()
            time -= 0.05
            # input()

        print("Cube Reached")
        
        q = [0]*len(q)
        self.robot.set_joint_target_velocities(q)
        grasped = False
        while not grasped:
            grasped = self.bot.close_gripper()
            self.pr.step()
            # print(grasped)
            # print(self.bot.gripper.get_joint_forces())
        print("Cube Grasped")
        self.stayStill(0.1)
        # print(f"Last force: {self.bot.gripper.get_joint_forces()}")

        destination = self.bot.getTip().get_position()
        destination[2] += 0.3
        self.displaceArm(destination)

        self.curve.remove_dummies()
        dmove.remove_dummy()

    def check_cubeInDistance(self, target_dist=0.04) -> bool:
        distance = self.target.get_position() - self.robot._ik_tip.get_position()
        distance = np.linalg.norm(distance)
        return True if distance <= target_dist else False

    def check_cubeReached(self, threshold=0.04) -> bool:
        distance = self.target.get_position(relative_to=self.bot.getTip())
        front_distance = distance[0]
        # print(f"Front distance: {front_distance}")
        lateral_distance = np.abs(distance[1])
        if lateral_distance >= 0.015: #check that cube is between gripper fingers
            return False
        elif front_distance >= 0 and front_distance <= threshold: #check that cube is inside gripper
            return True
        else:
            # print("Was within fingers")
            return False

    def autonomousMovement(self, model: nn.Module, transform: T.Compose):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        # img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        img: torch.Tensor = transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        # print(model(img).shape)
        v: torch.Tensor = model(img)[0]
        v = v.detach().numpy()
        q = self.bot.get_jointVelo(v)
        self.robot.set_joint_target_velocities(q)
        self.pr.step()

    def autonomousMovement_constrained(self, model: nn.Module, transform: T.Compose):
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

    def autonomousStop(self, model: nn.Module, transform: T.Compose, constrained: bool=False):
        #take the image from the robot
        img = self.camera.capture_rgb()
        img = Image.fromarray(np.uint8(img*255)).convert('RGB')
        img: torch.Tensor = transform(img)
        img = img.unsqueeze(0)
        #shove it into the model
        # print(model(img).shape)
        v, stop = model(img)
        # print(out.shape)
        v = v[0].detach().numpy()
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

    # def humanMovement(self, time: float):
    #     theta = self.curve.find_middlePoint()
    #     dmove = DummyMovement(self.target, time) 
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance_vec = self.bot.get_movementDir(self.target)
    #     direction = distance_vec / np.linalg.norm(distance_vec)
    #     v_lin = (self.curve.get_arcLen()/time) * direction

    #     for i in range(n_steps):
    #         if i % 3 != 0:
    #             v = self.curve.getVelocity2Target(v_lin)
    #             q = self.bot.get_jointVelo(v)
    #         else:
    #             self.curve.getVelocity2Target(v_lin)
    #             orientation = self.curve.get_FaceTargetOrientation()
    #             w = self.bot.get_angularSpeed(orientation)
    #             q = self.bot.get_jointVelo_4orientation(w)
    #         self.robot.set_joint_target_velocities(q)
    #         self.pr.step()
    #         if self.check_cubeReached():
    #             print(f"Cube Reached at step {i+1}")
    #             break
    #     self.curve.remove_dummies()

    # def humanMovement(self, time: float):
    #     theta = self.curve.find_middlePoint()
    #     dmove = DummyMovement(self.target, time)
    #     n_steps = np.round(time / 0.05).astype(int)

    #     distance = self.bot.get_movementDir(self.target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (self.curve.get_arcLen()/time) * direction

    #     # print("cavolfiore: ", 13/n_steps)

    #     for i in range(n_steps):
    #         if i % 3 ==0:
    #             v = self.curve.getVelocity2Target(v_lin)
    #             q = self.bot.get_jointVelo(v)
    #         else:
    #             # orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
    #             orientation = self.curve.get_FaceTargetOrientation()
    #             # v = self.curve.get_tangentVelocity(v_lin)
    #             # v = self.curve.get_enhancedTangentVelocity(v_lin, time)
    #             v = self.curve.getVelocity2Target(v_lin)
    #             print(v)
    #             # input()
    #             w = self.bot.get_angularSpeed(orientation)
    #             # w = np.array([0,0,0])
    #             q = self.bot.get_jointVelo_constrained(v, w)
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