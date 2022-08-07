from typing import List, Tuple
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
import math

from .my_robot import MyRobot
from .custom_baxter import CustomBaxter


class BaxterBot(MyRobot):

    def __init__(self) -> None:
        robot = LBRIwaa14R820()
        gripper = CustomBaxter()

        super().__init__(robot, gripper)

    # Overriding abstract method
    def get_DefaultConfiguration(self) -> Tuple[List[float], List[float]]:
        robot_config = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
        aperture_range = self.gripper.get_joint_intervals()[1][0]
        # print(aperture_range)
        gripper_config = [aperture_range[0]] #, 0]
        # gripper_config = [-0.03, 0]
        return (robot_config, gripper_config)

    # Overriding abstract method
    def close_gripper(self) -> bool:
        gripper: CustomBaxter = self.gripper
        return gripper.holdTight(0.04)