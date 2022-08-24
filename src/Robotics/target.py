from typing import List
import numpy as np
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

class Target(Shape):

    def __init__(self, target_type: str='cube', size: List[float]=None, color: List[float]=None) -> None:
        handle, _height = self._create_target(target_type, size)
        super().__init__(handle)
        self._set_color(color)
        self.set_renderable(True)

        self.position_min, self.position_max = [-0.5, 0.5, _height/2], [0.5, 0.85, _height/2] #[-0.5, 0.5, 0.3], [0.5, 1, 0.3]
        self.initailOrientation = self.get_orientation()
        self.random_pos()

        # self.set_dynamic(False)
    
    def _create_target(self, target_type: str, size: List[float]):
        _type = self._get_pureShape(target_type)
        _options = 24 - 16 #static, responsable
        self._size = self._get_size(size)
        _height = self._size[2]
        _mass = 1.
        handle = sim.simCreatePureShape(_type.value, _options, self._size, _mass, None)
        return handle, _height

    def _get_pureShape(self, target_type: str):
        if target_type in ['cube', 'rectangle']:
            return PrimitiveShape.CUBOID
        elif target_type == 'cylinder':
            return PrimitiveShape.CYLINDER
        elif target_type == 'sphere':
            return PrimitiveShape.SPHERE
        else:
            raise Exception("The chosen target's shape is not supported")

    def _get_size(self, size: List[float]):
        if size is None:
            return [0.05, 0.05, 0.05]
        else:
            return size

    def _set_color(self, color: List[float]):
        if color is None:
            self.set_color([1.0, 0.1, 0.1]) #red
        else:
            self.set_color(color)

    def get_size(self):
        return self._size

    def set_posBoundaries(self, min, max):
        self.position_min, self.position_max = min, max

    def set_restrictedBoundaries(self, restriction_type: str="slightly"):
        if restriction_type == "None":
            pass
        elif restriction_type == "slightly":
            self.position_min = [-0.35, 0.55, 0.025]
            self.position_max = [0.35, 0.85, 0.025]
        elif restriction_type == "moderate":
            self.position_min = [-0.26, 0.63, 0.025]
            self.position_max = [0.26, 0.85, 0.025]
        elif restriction_type == "highly":
            self.position_min = [-0.20, 0.60, 0.025]
            self.position_max = [0.20, 0.80, 0.025]
        else:
            raise Exception("Cube restiction option is not available")
        
    def random_pos(self):
        pos = list(np.random.uniform(self.position_min, self.position_max))
        # pos = [0, 0.75, 0.025]
        self.set_position(pos)
        self.set_orientation(self.initailOrientation)

    def get_occupancy(self):
        side = self._size[0]
        margin = 0.1
        occupancy = np.array([side/2, side/2]) + margin
        pos = self.get_position()[:2]
        return pos-occupancy, pos+occupancy