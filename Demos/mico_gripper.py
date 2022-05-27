from pyrep.robots.end_effectors.gripper import Gripper


class MicoGripperComplete(Gripper):

    def __init__(self, count: int = 0):
        super().__init__(count, 'MicoHand',
                         ['MicoHand_joint1_finger1',
                          'MicoHand_joint1_finger3',
                          'MicoHand_joint2_finger1',
                          'MicoHand_joint2_finger3',])
