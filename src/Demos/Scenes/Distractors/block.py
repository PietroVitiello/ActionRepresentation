from typing import List
import numpy as np
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

from .distractor import Distractor

class Block(Distractor):

    def __init__(self) -> None:
        shape_type = PrimitiveShape.CUBOID
        color = [204, 0, 204] #purple
        side = 0.1
        length = 0.2
        size = [length, side, side]
        super().__init__(shape_type, size, color)

    def _define_bounding_box(self):
        angle = self.get_orientation()[2]
        side_x = self._size[0]
        side_y = self._size[1]
        x = side_x * abs(np.cos(angle)) + side_y * abs(np.sin(angle))
        y = side_x * abs(np.sin(angle)) + side_y * abs(np.cos(angle))
        occupancy = np.array([x/2, y/2])
        return occupancy

    def random_orientation(self):
        angle = np.random.uniform(0, 2*np.pi)
        orientation = self.get_orientation()
        orientation[2] = angle
        self.set_orientation(orientation)
        self.initailOrientation = orientation