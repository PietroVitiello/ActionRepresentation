from os.path import dirname, join, abspath
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable, Generator, Tuple

from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

from Robotics.Kinematics.quadratic import Quadratic
from Robotics.Robot.my_robot import MyRobot
from Robotics.target import Target
from Robotics.Kinematics.dummy_movement import DummyMovement

import time as t
import math

class DataGenerator():

    def __init__(self, pr: PyRep, bot: MyRobot, res: int= 64) -> None:
        self.pr = pr

        self.bot = bot #MyRobot()
        self.target = Target()
        self.camera = VisionSensor("Vision_sensor")
        self.curve = Quadratic(self.bot.robot.get_tip(), self.target)

        self.setCameraRes(res)
        self.time = 0
        self.stop = 0

    def setCameraRes(self, res: int) -> None:
        self.camera.set_resolution([res, res])

    def restrictTargetBound(self):
        self.target.set_restrictedBoundaries()

    def terminate(self) -> None:
        self.pr.stop()
        self.pr.shutdown()

    def setTime(self, time: float) -> None:
        self.time = time

    def simStep(self):
        self.time -= 0.05
        self.pr.step()

    def resetEpisode(self):
        self.target.random_pos()
        self.resetRun()

    def resetRun(self):
        self.bot.resetInitial(self.pr)
        self.stop = 0

    def get_CurrentData(self) -> Tuple:
        im = self.camera.capture_rgb()
        joint_vel = self.bot.robot.get_joint_velocities()
        joint_target_vel = self.bot.robot.get_joint_target_velocities()
        joint_pos = self.bot.robot.get_joint_positions()

        ee_pos = self.bot.robot.get_tip().get_position()
        ee_orientation = self.bot.robot.get_tip().get_orientation()
        ee_vel = np.concatenate(list(self.bot.robot.get_tip().get_velocity()), axis=0)

        cube_pos = self.target.get_position()
        rel_cubePos = cube_pos - ee_pos
        
        return (im, joint_target_vel, joint_vel, joint_pos, ee_vel, ee_pos, ee_orientation, rel_cubePos, self.stop)

    def check_cubeReached(self, threshold=0.04) -> bool:
        distance = self.target.get_position() - self.bot.robot._ik_tip.get_position()
        distance = np.linalg.norm(distance)
        return True if distance <= threshold else False

    def grasp(self):
        v = np.array([0,0,0])
        grasped = False
        while not grasped:
            grasped = self.bot.close_gripper()
            yield v, *self.get_CurrentData()
            self.pr.step()


    def moveArm(self, v_lin) -> None:
        distance = self.bot.get_movementDir(self.target)
        v = self.bot.get_linearVelo(distance, self.time)
        q = self.bot.get_jointVelo(v)
        self.bot.robot.set_joint_target_velocities(q)

    def moveArm_constrained(self, v_lin) -> None:
        distance = self.bot.get_movementDir(self.target)
        orientation = self.curve.linear_mid.get_orientation(relative_to=self.bot.robot._ik_tip)
        v = self.bot.get_linearVelo(distance, self.time)
        w = self.bot.get_angularSpeed(orientation)
        q = self.bot.get_jointVelo_constrained(v, w)
        self.bot.robot.set_joint_target_velocities(q)

    def moveArmCurved(self, v_lin) -> None:
        v = self.curve.get_tangentVelocity(v_lin)
        q = self.bot.get_jointVelo(v)
        self.bot.robot.set_joint_target_velocities(q)

    def moveArmCurved_constrained(self, v_lin) -> None:
        orientation = self.curve.get_FaceTargetOrientation()
        v = self.curve.get_tangentVelocity(v_lin)
        w = self.bot.get_angularSpeed(orientation)
        q = self.bot.get_jointVelo_constrained(v, w)
        self.bot.robot.set_joint_target_velocities(q)

    def setGenerator(self, movement_function: Callable):
        n_steps = np.round(self.time / 0.05).astype(int)
        self.curve.find_middlePoint()

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        for _ in range(n_steps):
            movement_function(v_lin)
            yield self.get_CurrentData()
            self.simStep()

        self.curve.remove_dummies()

    def linearTrjGenerator(self, distance2cube: float=0.03):
        while self.check_cubeReached(distance2cube) is False:
            distance = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.bot.robot._ik_tip)
            v = self.bot.get_linearVelo(distance, self.time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)
            yield np.hstack((v, w)), *self.get_CurrentData()
            self.simStep()

    # def humanTrjGenerator(self):
    #     self.curve.find_middlePoint()
    #     dmove = DummyMovement(self.target, self.time)
    #     n_steps = np.round(self.time / 0.05).astype(int)

    #     distance = self.bot.get_movementDir(self.target)
    #     direction = distance / np.linalg.norm(distance)
    #     v_lin = (self.curve.get_arcLen()/self.time) * direction

    #     while self.check_cubeReached() is False:
    #         orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
    #         v = self.curve.get_enhancedTangentVelocity(v_lin, self.time)
    #         w = self.bot.get_angularSpeed(orientation)
    #         q = self.bot.get_jointVelo_constrained(v, w)
    #         self.bot.robot.set_joint_target_velocities(q)
    #         yield self.get_CurrentData()
    #         self.simStep()
    #         dmove.step()
    #     self.curve.remove_dummies()
    #     dmove.remove_dummy()

    def imperfect_humanTrjGenerator(self, distance2cube: float=0.03):
        self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, self.time)
        n_steps = np.round(self.time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        for i in range(n_steps):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.get_enhancedTangentVelocity(v_lin, self.time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            if self.check_cubeReached(distance2cube):
                i = n_steps + 1
                q = np.zeros(q.shape)
                print("Cube reached")
            self.bot.robot.set_joint_target_velocities(q)
            yield np.hstack((v, w)), *self.get_CurrentData()
            self.simStep()
            dmove.step()
        self.curve.remove_dummies()
        dmove.remove_dummy()

    def humanTrjGenerator(self, distance2cube: float=0.03):
        self.curve.find_middlePoint()
        # dmove = DummyMovement(self.target, self.time)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        while self.check_cubeReached(distance2cube) is False:
            v = self.curve.getVelocity2Target(v_lin)
            q = self.bot.get_jointVelo(v)
            self.bot.robot.set_joint_target_velocities(q)
            yield v, *self.get_CurrentData()
            self.simStep()
        self.curve.remove_dummies()
        # dmove.remove_dummy()

    def humanTrjGenerator_fixedSteps(self):
        self.curve.find_middlePoint()
        # dmove = DummyMovement(self.target, self.time)
        n_steps = np.round(self.time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        for _ in range(n_steps):
            v = self.curve.getVelocity2Target(v_lin)
            q = self.bot.get_jointVelo(v)
            self.bot.robot.set_joint_target_velocities(q)
            yield v, *self.get_CurrentData()
            self.simStep()
        self.curve.remove_dummies()
        # dmove.remove_dummy()

    def humanTrjGenerator_stop(self):
        self.curve.find_middlePoint()

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        while self.check_cubeReached(0.01) is False:
            v = self.curve.getVelocity2Target(v_lin)
            q = self.bot.get_jointVelo(v)
            self.bot.robot.set_joint_target_velocities(q)
            yield v, *self.get_CurrentData()
            self.simStep()

        v = np.array([0,0,0])
        q = [0]*len(q)
        self.bot.robot.set_joint_target_velocities(q)
        self.stop = 1
        self.curve.remove_dummies()
        yield v, *self.get_CurrentData()

    def getLinearTrjGenerator(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        return self.linearTrjGenerator(distance2cube)

    def getHumanTrjGenerator_imperfect(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        return self.imperfect_humanTrjGenerator(distance2cube)

    def getHumanTrjGenerator(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        return self.humanTrjGenerator(distance2cube)

    def getHumanTrjGenerator_fixedSteps(self, time: float=2) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        return self.humanTrjGenerator_fixedSteps()

    def getHumanTrjGenerator_stop(self, time: float=2) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        distance2cube = 0.01
        return distance2cube, self.humanTrjGenerator_stop()

    def getGenerator(self, move_type: str, constraint: str, time: float=2) -> Generator:
        if move_type == 'human-like':
            pass
        elif constraint == 'normal':
            if move_type == 'linear':
                movement = self.moveArm
            elif move_type == 'curved':
                movement = self.moveArmCurved
            else:
                print("The type of movement can either be 'linear' or 'curved'")
                return 0

        elif constraint == 'constrained':
            if move_type == 'linear':
                movement = self.moveArm_constrained
            elif move_type == 'curved':
                movement = self.moveArmCurved_constrained
            else:
                print("The type of movement can either be 'linear' or 'curved'")
                return 0

        else:
            print("The constraint can either be 'normal' or 'constrained'")
            return 0

        self.setTime(time)
        self.curve.resetCurve()
        return self.setGenerator(movement)
