from pyrep.objects.dummy import Dummy
import numpy as np
import math

class Circle():

    def __init__(self, tip, target, max_deviation=0.05) -> None:
        self.tip = tip
        self.target = target
        self.max_deviation = max_deviation

        self.distance = self.target - self.tip
        linear_mid_pos = (target + tip)/2
        self.linear_mid = Dummy.create()
        self.set_linearMid(linear_mid_pos)

    def set_linearMid(self, pos):
        self.linear_mid.set_position(pos)
        direction = self.distance / np.linalg.norm(self.distance)
        gamma = np.arctan2(direction[1], direction[0])
        beta = - np.arctan2(direction[2], direction[0])

    def find_middlePoint(self):
        deviation = np.random.uniform(self.max_deviation)
        theta = np.random.uniform(-180, 180)



        
