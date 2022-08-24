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

    def _environment_startup(self):
        self.reset_scene

    def reset_scene(self):
        self.target.random_pos()
        
        