import numpy as np

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

    def get_distractors(self):
        return self.distractors

    def get_distractors_state(self, as_concatenated_array=False):
        get_position = lambda obj: np.concatenate((obj.get_position(), [obj.get_orientation()[2]]))
        distr_positions = list(map(get_position, self.distractors))
        if as_concatenated_array is False:
            return distr_positions
        else:
            return np.concatenate(distr_positions)

    def _environment_startup(self):
        self.spawn_distractors()

    def reset_scene(self):
        for distractor in self.distractors:
            distractor: Distractor
            distractor.remove()
        self.distractors = []
        self.target.random_pos()
        self.spawn_distractors()

    def set_scene(self, data):
        self.reset_scene()
        self.target : Target
        self.target.set_position(data[:3])
        for i in range(self.n_distractors):
            obj: Distractor = self.distractors[i]
            index = (i * 4) + 3
            obj.set_position(data[index:index+3])
            orientation = obj.get_orientation()
            orientation[2] = data[index+3]
            # obj.set_orientation(orientation)

        
        