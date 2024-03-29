import numpy as np

from .scene import Scene
from Robotics.target import Target

from .Distractors.distractor import Distractor
from .Distractors.block import Block
from .Distractors.cone import Cone
from .Distractors.cylinder import Cylinder

class Shape_Scene(Scene):
    def __init__(
        self,
        boundary_restriction: str = 'moderate',
        shape: str = 'cube'
    ) -> None:
        super().__init__()
        self.boundary_restriction = boundary_restriction
        self.shape = shape
        self.supported_shapes = [
            'cube',
            'vertical rectangle',
            'horizontal rectangle',
            'cylinder',
            'tall cylinder'   
        ]
        self.distance2cube = None

    def get_supported_shapes(self):
        return self.supported_shapes

    def get_distance2cube(self):
        return self.distance2cube

    def shape_shift_target(self, shape, size, shape_id):
        self.target.remove()
        self.target = Target(shape, size)
        self.target.set_restrictedBoundaries(self.boundary_restriction)
        self.reset_scene()
        self.shape = shape_id

    def make_cube(self):
        shape = 'cube'
        size = None
        self.shape_shift_target(shape, size, shape_id=0)
        self.distance2cube = None

    def make_vertical_rectangle(self):
        shape = 'rectangle'
        size = [0.05, 0.05, 0.12]
        self.shape_shift_target(shape, size, shape_id=1)
        self.distance2cube = None

    def make_horizontal_rectangle(self):
        shape = 'rectangle'
        size = [0.05, 0.12, 0.05]
        self.shape_shift_target(shape, size, shape_id=2)
        self.distance2cube = None

    def make_cylinder(self):
        shape = 'cylinder'
        size = [0.05, 0.05, 0.05]
        self.shape_shift_target(shape, size, shape_id=3)
        self.distance2cube = 0.001

    def make_tall_cylinder(self):
        shape = 'cylinder'
        size = [0.05, 0.05, 0.12]
        self.shape_shift_target(shape, size, shape_id=4)
        self.distance2cube = 0.001

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
        self.reset_scene()

    def reset_scene(self):
        self.target.random_pos()

    def set_scene(self, data):
        shape_index = int(data[-1])
        shape = self.supported_shapes[shape_index]
        self.change_shape(shape)
        self.target.set_position(data[:3])

    def log_test_run(self, run, n_completions):
        run.log({
            f"grasped_{get_id_name(self.shape)}": n_completions
        })



def get_id_name(index):
    names = [
        'cubes',
        'vertical_rectangles',
        'horizontal_rectangles',
        'cylinders',
        'tall_cylinders'   
    ]
    return names[index]
        