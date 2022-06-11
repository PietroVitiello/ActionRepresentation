import numpy as np
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

class Target(Shape):

    def __init__(self) -> None:
        _type = PrimitiveShape.CUBOID
        _options = 24 - 16#static, responsable
        _size = [0.05, 0.05, 0.05]
        _mass = 1.
        handle = sim.simCreatePureShape(_type.value, _options, _size, _mass, None)
        super().__init__(handle)
        self.set_color([1.0, 0.1, 0.1])
        self.set_renderable(True)

        self.position_min, self.position_max = [-0.5, 0.5, 0.025], [0.5, 0.85, 0.025] #[-0.5, 0.5, 0.3], [0.5, 1, 0.3]
        self.initailOrientation = self.get_orientation()
        self.random_pos()

        # self.set_dynamic(False)

    def set_posBoundaries(self, min, max):
        self.position_min, self.position_max = min, max

    def set_restrictedBoundaries(self):
        self.position_min = [-0.35, 0.55, 0.025]
        self.position_max = [0.35, 0.85, 0.025]

    def random_pos(self):
        # pos = list(np.random.uniform(self.position_min, self.position_max))
        pos = [0, 0.75, 0.025]
        self.set_position(pos)
        self.set_orientation(self.initailOrientation)