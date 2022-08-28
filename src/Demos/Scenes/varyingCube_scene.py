import numpy as np

from .scene import Scene
from Robotics.target import Target

from .Distractors.distractor import Distractor
from .Distractors.block import Block
from .Distractors.cone import Cone
from .Distractors.cylinder import Cylinder

class VaryingCube_Scene(Scene):
    def __init__(
        self,
        boundary_restriction,
        shape: str = 'cube'
    ) -> None:
        super().__init__()
        self.boundary_restriction = boundary_restriction
        self.shape = shape
        self.supported_shapes = [
            'small',
            'regular',
            'large' 
        ]

    def get_supported_shapes(self):
        return self.supported_shapes

    def shape_shift_target(self, shape, size):
        self.target.remove()
        self.target = Target(shape, size)
        self.target.set_restrictedBoundaries(self.boundary_restriction)
        self.reset_scene()
        self.shape = shape

    def make_cube(self):
        shape = 'cube'
        size = None
        self.shape_shift_target(shape, size)

    def make_vertical_rectangle(self):
        shape = 'rectangle'
        size = [0.05, 0.05, 0.12]
        self.shape_shift_target(shape, size)

    def make_horizontal_rectangle(self):
        shape = 'rectangle'
        size = [0.05, 0.12, 0.05]
        self.shape_shift_target(shape, size)

    def make_cylinder(self):
        shape = 'cylinder'
        size = [0.05, 0.05, 0.05]
        self.shape_shift_target(shape, size)

    def make_tall_cylinder(self):
        shape = 'cylinder'
        size = [0.05, 0.05, 0.12]
        self.shape_shift_target(shape, size)

    def change_shape(self, shape):
        if shape == 'cube':
            self.make_cube()
        elif shape == 'vertical rectangle':
            self.make_vertical_rectangle()
        elif shape == 'horizontal rectangle':
            self.make_horizontal_rectangle()
        elif shape == 'cylinder':
            self.make_cylinder()
        elif shape == 'tall cylinder':
            self.make_tall_cylinder()
        else:
            raise Exception("Chosen shape for the target is not supported yet")

    def next_shape(self, i):
        possibilities = len(self.supported_shapes)
        i = i % possibilities
        next_shape = self.supported_shapes[i]
        self.change_shape(next_shape)

    def set_target_object(self, target):
        self.target = target
        self.change_shape(self.shape)

    def _environment_startup(self):
        self.reset_scene

    def reset_scene(self):
        self.target.random_pos()

    def set_scene(self, data):
        print(data)
        self.target : Target
        data = np.concatenate(data)
        self.target.set_position(data)
        