from .scene import Scene
from Robotics.target import Target

from .Distractors.distractor import Distractor
from .Distractors.block import Block
from .Distractors.cone import Cone
from .Distractors.cylinder import Cylinder

class Distractor_Scene(Scene):
    def __init__(
        self,
        n_distractors: int
    ) -> None:
        self.n_distractors = n_distractors
        self.distractors = []
        super().__init__()

    def spawn_distractors(self):
        distractors = [Cone, Block, Cylinder]
        objects_in_scene = [self.target]
        for obj in distractors[:self.n_distractors]:
            distractor: Distractor = obj()
            distractor.random_pos(objects_in_scene)
            objects_in_scene.append(distractor)
            self.distractors.append(distractor)

    def _environment_startup(self):
        self.spawn_distractors()

    def reset_scene(self):
        for distractor in self.distractors:
            distractor: Distractor
            distractor.remove()
        self.distractors = []
        self.target.random_pos()
        self.spawn_distractors()

    def get_distractors(self):
        return self.distractors
        
        