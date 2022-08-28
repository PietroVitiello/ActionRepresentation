from __future__ import annotations
from abc import abstractclassmethod
from typing import List, Union
import numpy as np
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

from Robotics.target import Target

class Distractor(Shape):

    def __init__(
        self,
        shape_type,
        size,
        color
    ) -> None:
        options = 24 - 16 #static, responsable
        self._size = size
        height = self._size[2]
        mass = 1.
        handle = sim.simCreatePureShape(shape_type.value, options, self._size, mass, None)

        super().__init__(handle)
        self._set_color(color) 
        self.set_renderable(True)
        self.set_collidable(True)

        self.position_min, self.position_max = [-0.4, 0.6, height/2], [0.4, 0.85, height/2] #[-0.26, 0.63, 0.025] [0.26, 0.85, 0.025]
        self.initailOrientation = self.get_orientation()
        self._set_far_position()
        self.random_orientation()

        # self.set_dynamic(False)

    def _set_color(self, color: List[float]):
        color = np.array(color)/255
        self.set_color(list(color))

    def _set_far_position(self):
        height = self._size[2]
        pos = [20, 20, height/2]
        self.set_position(pos)

    def get_size(self):
        return self._size

    @abstractclassmethod
    def _define_bounding_box(self):
        pass

    def get_occupancy(self, pos=None):
        if pos is None:
            pos = self.get_position()
        margin = 0.03
        occupancy = self._define_bounding_box() + margin
        pos = pos[:2]
        return pos-occupancy, pos+occupancy

    def set_posBoundaries(self, min, max):
        self.position_min, self.position_max = min, max

    # def set_restrictedBoundaries(self, restriction_type: str="slightly"):
    #     if restriction_type == "None":
    #         pass
    #     elif restriction_type == "slightly":
    #         self.position_min = [-0.35, 0.55, 0.025]
    #         self.position_max = [0.35, 0.85, 0.025]
    #     elif restriction_type == "moderate":
    #         self.position_min = [-0.30, 0.60, 0.025]
    #         self.position_max = [0.30, 0.85, 0.025]
    #     elif restriction_type == "highly":
    #         self.position_min = [-0.20, 0.60, 0.025]
    #         self.position_max = [0.20, 0.80, 0.025]
    #     else:
    #         raise Exception("Cube restiction option is not available")
        

    def random_pos(self, other_objects: List[Union[Target, Distractor]] = None):
        valid_pos = False
        n_iterations = 0
        while valid_pos is False:
            valid_pos = True

            if n_iterations == 300:
                for i, obj in enumerate(other_objects[1:]):
                    obj.random_pos(other_objects[:i+1])
                n_iterations = 0
            
            pos = list(np.random.uniform(self.position_min, self.position_max))
            for obj in other_objects:
                min_pos, max_pos = self.get_occupancy(pos)
                other_min_pos, other_max_pos = obj.get_occupancy()
                if check_pos_outside(min_pos, max_pos, other_min_pos, other_max_pos) is False:
                    valid_pos = False
                    break
            n_iterations += 1

        self.set_position(pos)
        self.set_orientation(self.initailOrientation)

    @abstractclassmethod
    def random_orientation(self):
        pass

def check_pos_outside(minA, maxA, minB, maxB):
    outside = False
    if maxA[0] < minB[0] or minA[0] > maxB[0]:
        outside = True
    elif maxA[1] < minB[1] or minA[1] > maxB[1]:
        outside = True
    return outside