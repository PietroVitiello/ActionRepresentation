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

from .Scenes.scene import Scene
import time as t
import math

class DataGenerator():

    def __init__(
        self,
        pr: PyRep,
        scene: Scene,
        bot: MyRobot,
        res: int= 64,
        max_deviation: float=0.04,
        always_maxDev: bool=True,
    ) -> None:
        self.pr = pr

        self.bot = bot #MyRobot()
        self.target = Target()
        self.camera = VisionSensor("Vision_sensor")
        self.curve = Quadratic(self.bot.robot.get_tip(), self.target, max_deviation, always_maxDev)

        self.scene = scene
        self.scene.set_target_object(self.target)
        try:
            self.ep_pos = self.target.get_position()
            self.ep_orientation = self.target.get_orientation()
        except RuntimeError:
            pass

        self.setCameraRes(res)
        self.time = 0
        self.stop = 0
        self.grasped = False

    def setCameraRes(self, res: int) -> None:
        self.camera.set_resolution([res, res])

    def set_target(self, target):
        self.target = target
        self.curve.setTarget(self.target)

    def restrictTargetBound(self, restriction_type: str):
        self.scene.restrictTargetBound(restriction_type)

    def terminate(self) -> None:
        self.pr.stop()
        self.pr.shutdown()

    def setTime(self, time: float) -> None:
        self.time = time

    def simStep(self):
        self.time -= 0.05
        self.pr.step()

    def resetEpisode(self):
        self.scene.reset_scene()
        self.ep_pos = self.target.get_position()
        self.ep_orientation = self.target.get_orientation()
        self.resetRun()

    def resetRun(self):
        self.bot.resetInitial(self.pr)
        self.target.set_position(self.ep_pos)
        self.target.set_orientation(self.ep_orientation)
        self.stop = 0
        self.grasped = False

    def get_CurrentData(self) -> Tuple:
        im = self.camera.capture_rgb()
        joint_vel = self.bot.robot.get_joint_velocities()
        joint_target_vel = self.bot.robot.get_joint_target_velocities()
        joint_pos = self.bot.robot.get_joint_positions()

        ee_pos = self.bot.robot.get_tip().get_position()
        ee_orientation = self.bot.robot.get_tip().get_orientation()
        ee_vel = np.concatenate(list(self.bot.robot.get_tip().get_velocity()), axis=0)

        cube_pos = self.target.get_position()
        rel_cubePos = cube_pos - ee_pos #relative is hard to calculate in real setting (I discarded it)
        
        return (im, joint_target_vel, joint_vel, joint_pos, ee_vel, ee_pos, ee_orientation, rel_cubePos, self.stop)

    def get_cube_pos(self):
        return self.target.get_position()

    def check_cubeInDistance(self, threshold_dist=0.04) -> bool:
        distance = self.target.get_position() - self.bot.robot._ik_tip.get_position()
        distance = np.linalg.norm(distance)
        return True if distance <= threshold_dist else False
        
    def check_cubeReached(self, threshold=0.04) -> bool:
        distance = self.target.get_position(relative_to=self.bot.getTip())
        front_distance =  distance[0]
        lateral_distance = np.abs(distance[1])
        if lateral_distance >= 0.015: #check that cube is between gripper fingers
            return False
        elif front_distance >= 0 and front_distance <= threshold: #check that cube is inside gripper
            return True
        return False

    def checkCubeGrasped(self):
        cube_elevation = self.target.get_position()[2]
        side = self.target.get_size()[0]
        return cube_elevation > 2*side

    def grasp(self, dof: int=6):
        v = np.array([0]*dof)
        grasped = False
        while not grasped:
            grasped = self.bot.close_gripper()
            yield self.grasped, (v, *self.get_CurrentData())
            self.pr.step()

    def lift_grasped(self, displacement: float=0.25, time: float=2):
        destination = self.bot.getTip().get_position()
        destination[2] += displacement
        n_steps = np.round(time / 0.05).astype(int)
        self.setTime(time)
        for _ in range(n_steps):
            distance_vec = destination - self.bot.getTip().get_position()
            v = self.bot.get_linearVelo(distance_vec, self.time)
            q = self.bot.get_jointVelo(v)
            self.bot.robot.set_joint_target_velocities(q)
            yield self.grasped, (v, *self.get_CurrentData())
            self.simStep()

    def stayStill(self, time: float):
        self.bot.robot.set_joint_target_velocities([0]*7)
        n_steps = np.round(time / 0.05).astype(int)
        for _ in range(n_steps):
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

    ############### Generators ###############

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
        reached = False
        while reached is False:
            distance = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.bot.robot._ik_tip)
            v = self.bot.get_linearVelo(distance, self.time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)
            eeVel, data = np.hstack((v, w)), self.get_CurrentData()
            self.simStep()
            reached = self.check_cubeReached(distance2cube)
            yield reached, (eeVel, *data)
        print(f"\033[32mCube Reached\033[37m", end="")

    def humanTrjGenerator(self, distance2cube: float=0.03):
        self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, self.time, tip=self.bot.getTip())
        n_steps = np.round(self.time / 0.05).astype(int)
        # n_steps = int(n_steps*1.1) #REMOVE

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        for i in range(n_steps):
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.getVelocity2Target(v_lin)
            w = self.bot.get_angularSpeed(orientation, time=0.05)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)

            # yield np.hstack((v, w)), *self.get_CurrentData()
            eeVel = np.hstack((v, w))
            data = self.get_CurrentData()
            self.simStep()
            dmove.step()
            yield (self.check_cubeReached(distance2cube), (eeVel, *data))

            if self.check_cubeReached(distance2cube):
                print(f"\033[32mCube Reached at step {i+1}\033[37m", end="")
                break
        self.curve.remove_dummies()
        dmove.remove_dummy()

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
            v = self.curve.get_tangentVelocity(v_lin)
            # v = self.curve.get_enhancedTangentVelocity(v_lin, self.time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)
            eeVel, data = np.hstack((v, w)), *self.get_CurrentData()
            self.simStep()
            dmove.step()
            reached = self.check_cubeReached(distance2cube)
            yield reached, (eeVel, *data)
            if reached:
                print(f"\033[32mCube Reached at step {i+1}\033[37m", end="")
                break
        self.curve.remove_dummies()
        dmove.remove_dummy()

    def followDummyGenerator(self, distance2cube: float=0.03):
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

    def followDummyGenerator_fixedSteps(self):
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

    def followDummyGenerator_stop(self):
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

    ############### Grasping Generators ###############

    def linearGrasp_generator(self, distance2cube: float):
        n_steps = np.round(self.time / 0.05).astype(int)

        for i in range(n_steps):
            distance = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.bot.robot._ik_tip)
            v = self.bot.get_linearVelo(distance, self.time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)
            eeVel, data = np.hstack((v, w)), self.get_CurrentData()
            self.simStep()
            yield self.grasped, (eeVel, *data)
            if self.check_cubeReached(distance2cube):
                print(f"\033[32mCube Reached at step {i+1}\033[38;5;231m", end=", ")
                break
        # print(f"Cube Reached", end="")
        q = [0]*len(q)
        self.bot.robot.set_joint_target_velocities(q)
        self.stop = 1

        for data in self.grasp(dof=6):
            yield data

        self.stayStill(0.3)
        for data in self.lift_grasped():
            yield data
        if self.checkCubeGrasped():
            self.grasped = True
            yield self.grasped, (np.array([0]), *self.get_CurrentData())
            print("\033[32mCube Grasped\033[38;5;231m", end="")
        
        self.curve.remove_dummies()

    def antiCollisionGrasp_generator(self, distance2cube: float):
        def check_distractor_collisions():
            collision = False
            for handle in distractors_handles:
                collision = True if self.bot.gripper.check_collision(handle) else collision
                collision = True if self.bot.robot.check_collision(handle) else collision
            # if collision: print("\033[31mOuch collision\033[38;5;231m", end="...")
            return collision

        n_steps = np.round(self.time / 0.05).astype(int)
        distractors_handles = self.scene.get_distractors()
        collided = False

        for i in range(n_steps):
            distance = self.bot.get_movementDir(self.target)
            orientation = self.curve.linear_mid.get_orientation(relative_to=self.bot.robot._ik_tip)
            v = self.bot.get_linearVelo(distance, self.time)
            w = self.bot.get_angularSpeed(orientation)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)
            eeVel, data = np.hstack((v, w)), self.get_CurrentData()
            self.simStep()
            collided = check_distractor_collisions()
            yield self.grasped, (eeVel, *data)
            if self.check_cubeReached(distance2cube):
                print(f"\033[32mCube Reached at step {i+1}\033[38;5;231m", end=", ")
                break
        # print(f"Cube Reached", end="")
        q = [0]*len(q)
        self.bot.robot.set_joint_target_velocities(q)
        self.stop = 1

        for data in self.grasp(dof=6):
            yield data

        self.stayStill(0.3)
        for data in self.lift_grasped():
            yield data
        if self.checkCubeGrasped():
            self.grasped = True
            success = (self.grasped and (not collided))
            yield success, (np.array([0]), *self.get_CurrentData())
            print("\033[32mCube Grasped\033[38;5;231m", end="")
        
        self.curve.remove_dummies()


    def humanTrjGrasp_generator(self, distance2cube: float):
        self.curve.find_middlePoint()
        dmove = DummyMovement(self.target, self.time, tip=self.bot.getTip())
        n_steps = np.round(self.time / 0.05).astype(int)

        distance = self.bot.get_movementDir(self.target)
        direction = distance / np.linalg.norm(distance)
        v_lin = (self.curve.get_arcLen()/self.time) * direction

        for i in range(n_steps):
        # while self.check_cubeReached(distance2cube) is False:
            orientation = self.curve.get_FaceTargetOrientation(dmove.getDummy())
            v = self.curve.getVelocity2Target(v_lin)
            w = self.bot.get_angularSpeed(orientation, time=0.05)
            q = self.bot.get_jointVelo_constrained(v, w)
            self.bot.robot.set_joint_target_velocities(q)
            yield np.hstack((v, w)), *self.get_CurrentData()
            self.simStep()
            dmove.step()

            # print(self.check_cubeReached(distance2cube))

            if self.check_cubeReached(distance2cube):
                print(f"\nCube Reached at step {i+1}")
                break
        # print(f"Cube Reached", end="")
        q = [0]*len(q)
        self.bot.robot.set_joint_target_velocities(q)
        self.stop = 1

        for data in self.grasp(dof=6):
            yield data
        print("Cube Grasped")

        self.stayStill(0.3)
        for data in self.lift_grasped():
            yield data
        
        self.curve.remove_dummies()
        dmove.remove_dummy()

    ############### Get Generators ###############

    def getLinearTrjGenerator(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        constrained = True #whether ee has also orientation constraint
        return constrained, self.linearTrjGenerator(distance2cube)

    def getHumanTrjGenerator(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        constrained = True #whether ee has also orientation constraint
        return constrained, self.humanTrjGenerator(distance2cube)

    def getHumanTrjGenerator_imperfect(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        constrained = True #whether ee has also orientation constraint
        return constrained, self.imperfect_humanTrjGenerator(distance2cube)

    def getLinearGraspGenerator(self, time: float=2, distance2cube=None) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        if distance2cube == None:
            distance2cube = 0.015 #0.025
        constrained = True #whether ee has also orientation constraint
        return distance2cube, constrained, self.linearGrasp_generator(distance2cube)

    def getHumanTrjGraspGenerator(self, time: float=2) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        distance2cube = 0.025
        constrained = True #whether ee has also orientation constraint
        return distance2cube, constrained, self.humanTrjGrasp_generator(distance2cube)

    def getFollowDummyGenerator(self, time: float=2, distance2cube: float=0.03) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        constrained = False #whether ee has also orientation constraint
        return constrained, self.followDummyGenerator(distance2cube)

    def getFollowDummyGenerator_fixedSteps(self, time: float=2) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        constrained = False #whether ee has also orientation constraint
        return constrained, self.followDummyGenerator_fixedSteps()

    def getFollowDummyGenerator_stop(self, time: float=2) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        distance2cube = 0.01
        constrained = False #whether ee has also orientation constraint
        return distance2cube, constrained, self.followDummyGenerator_stop()

    def getAntiCollisionGraspGenerator(self, time: float=2) -> Generator:
        self.setTime(time)
        self.curve.resetCurve()
        distance2cube = 0.015 #0.025
        constrained = True #whether ee has also orientation constraint
        return distance2cube, constrained, self.antiCollisionGrasp_generator(distance2cube)

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
