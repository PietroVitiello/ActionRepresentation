from typing import List, Tuple
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
import math

from .my_robot import MyRobot


class MicoBot(MyRobot):

    def __init__(self) -> None:
        robot = LBRIwaa14R820()
        gripper = MicoGripper()

        super().__init__(robot, gripper)

    # Overriding abstract method
    def get_DefaultConfiguration(self) -> Tuple[List[float], List[float]]:
        # initialConf = [0, math.radians(-30), 0, math.radians(-50), 0, math.radians(20), 0]
        # initialConf = [0, 0, 0, math.radians(-90), 0, math.radians(0), 0]
        # initialConf = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
        robot_config = [0, math.radians(-40), 0, math.radians(-130), 0, math.radians(60), 0]
        gripper_config = [0, 0]
        return (robot_config, gripper_config)