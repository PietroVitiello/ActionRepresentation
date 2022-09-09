import numpy as np

from .scene import Scene
from Robotics.target import Target

from .Distractors.distractor import Distractor
from .Distractors.block import Block
from .Distractors.cone import Cone
from .Distractors.cylinder import Cylinder

class Cube_Scene(Scene):
    def __init__(
        self
    ) -> None:
        super().__init__()

    def set_target_object(self, target):
        # super().set_target_object(target)
        self.target = target
        self._environment_startup()

    def _environment_startup(self):
        self.reset_scene()

    def reset_scene(self):
        self.target.random_pos()

    def set_scene(self, data):
        self.target.set_position(data)

    def log_test_run(self, run, n_completions):
        run.log({
            f"cubes_grasped": n_completions
        })
        