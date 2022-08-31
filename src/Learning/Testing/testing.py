from os.path import dirname, join, abspath
from typing import Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor

from Robotics.Robot.micoBot import MicoBot
from Robotics.Robot.baxterBot import BaxterBot
from Robotics.Robot.my_robot import MyRobot
from Robotics.target import Target
from Robotics.Kinematics.autonomous_rmove import Autonomous_RobotMovement

from Demos.Scenes.scene import Scene

class Test():

    def __init__(
        self,
        pr: PyRep,
        scene: Scene,
        model: nn.Module,
        data_transforms: T.Compose,
        restriction_type: str,
        camera_res=64,
        num_episodes = 32,
        max_n_steps = 140,
        saved_positions: pd.DataFrame = None
    ) -> None:

        self.pr = pr
        self.scene = scene

        self.data_transforms = data_transforms
        self.model = model

        # arm = get_arm()
        self.bot = BaxterBot()
        self.target = Target()
        camera = VisionSensor("Vision_sensor")
        self.rmove = Autonomous_RobotMovement(self.bot, self.target, self.pr, self.data_transforms, camera=camera, res=camera_res)
        self.target.set_restrictedBoundaries(restriction_type)
        self.scene.set_target_object(self.target)

        self.num_episodes = num_episodes
        self.max_n_steps = max_n_steps

        self.saved_positions = saved_positions.to_numpy()

    def parse_saved_positions(self):
        for row in self.saved_positions:
            self.scene.set_scene(row[1:])
            self.target = self.scene.get_target()
            yield 0

    def checkCubeGrasped(self):
        cube_elevation = self.target.get_position()[2]
        side = self.target.get_size()[0]
        return cube_elevation > 2*side

    def test_eeVel(self, constrained=False):
        if constrained:
            move: Callable = self.rmove.autonomousMovement_constrained
        else:
            move: Callable = self.rmove.autonomousMovement
        num_reached = 0
        for episode in range(self.num_episodes):
            print(f"Beginning episode {episode+1}")
            self.target.random_pos()
            self.bot.resetInitial(self.pr)
            self.rmove.stayStill(2)
            reached = False
            step_n = 0
            while reached == False and step_n<self.max_n_steps:
                move(self.model)
                reached = self.rmove.check_cubeReached(0.025)
                step_n += 1
                # print(step_n+1)
                # print(reached)
                # print(self.max_n_steps)
                # print(step_n<self.max_n_steps)
            if reached:
                print("Cube reached!\n")
                num_reached += 1
            else:
                print("Cube not reached\n")
            self.rmove.curve.remove_dummies()

        print(f"The robot was able to reach {num_reached} targets out of {self.num_episodes}")
        return num_reached

    def test_eeVel_LSTM(self):
        num_reached = 0
        for episode in range(self.num_episodes):
            print(f"Beginning episode {episode+1}")
            self.model.start_newSeq()
            self.target.random_pos()
            self.bot.resetInitial(self.pr)
            self.rmove.stayStill(2)
            reached = False
            step_n = 0
            while reached == False and step_n<self.max_n_steps:
                self.rmove.autonomousMovement(self.model)
                reached = self.rmove.check_cubeReached(0.04)
                step_n += 1

            if reached:
                print("Cube reached!\n")
                num_reached += 1
            else:
                print("Cube not reached\n")
            self.rmove.curve.remove_dummies()

        print(f"The robot was able to reach {num_reached} targets out of {self.num_episodes}")
        return num_reached

    def test_eeVelGrasp(self, constrained):
        num_reached = 0
        for episode in range(self.num_episodes):
            print(f"Beginning episode {episode+1}")
            self.target.random_pos()
            self.bot.resetInitial(self.pr)
            self.rmove.stayStill(2)
            stop = 0
            step_n = 0
            while stop < 0.96 and step_n<self.max_n_steps:
                stop = self.rmove.autonomousStop(self.model, constrained)
                step_n += 1

            print("Arm Thinks the Cube is Reached")

            n_joints = self.bot.robot.get_joint_count()
            q = [0]*n_joints
            self.bot.robot.set_joint_target_velocities(q)
            closed = False
            print("Attempting to Grasp Cube")
            while not closed:
                closed = self.bot.close_gripper()
                self.pr.step()

            destination = self.bot.getTip().get_position()
            destination[2] += 0.3
            self.rmove.displaceArm(destination)
            grasped = self.checkCubeGrasped()

            if grasped:
                print("Cube Grasped!\n")
                num_reached += 1
            else:
                print("Cube not Grasped\n")

            self.rmove.curve.remove_dummies()

        print(f"The robot was able to grasp {num_reached} targets out of {self.num_episodes}")
        return num_reached

    def test_eeVelGrasp_savedPos(self, constrained):
        num_reached = 0
        for ep, _ in enumerate(self.parse_saved_positions()):
            print(f"Beginning episode {ep+1}")
            self.target.set_orientation([0,0,0])
            self.bot.resetInitial(self.pr)
            stop = 0
            step_n = 0
            while stop < 0.96 and step_n<self.max_n_steps:
                stop = self.rmove.autonomousStop(self.model, constrained)
                step_n += 1

            print("Arm Thinks the Cube is Reached")

            n_joints = self.bot.robot.get_joint_count()
            q = [0]*n_joints
            self.bot.robot.set_joint_target_velocities(q)
            closed = False
            print("Attempting to Grasp Cube")
            while not closed:
                closed = self.bot.close_gripper()
                self.pr.step()

            destination = self.bot.getTip().get_position()
            destination[2] += 0.3
            self.rmove.displaceArm(destination)
            grasped = self.checkCubeGrasped()

            if grasped:
                print("Cube Grasped!\n")
                num_reached += 1
            else:
                print("Cube not Grasped\n")

            self.rmove.curve.remove_dummies()

        print(f"The robot was able to grasp {num_reached} targets out of {self.num_episodes}")
        return num_reached