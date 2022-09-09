from typing import List
import numpy as np
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

from .distractor import Distractor

class Cone(Distractor):

    def __init__(self) -> None:
        shape_type = PrimitiveShape.CONE
        color = [0, 204, 0] #green
        diameter = 0.1
        height = 0.25
        size = [diameter, diameter, height]
        super().__init__(shape_type, size, color)

    def _define_bounding_box(self):
        side = self._size[0]
        occupancy = np.array([side/2, side/2])
        return occupancy

    def random_orientation(self):
        pass