from abc import abstractclassmethod
from Robotics.target import Target

class Scene():
    def __init__(
        self,
    ) -> None:
        pass

    def set_target_object(self, target):
        self.target = target
        self._environment_startup()

    @abstractclassmethod
    def _environment_startup(self):
        pass

    @abstractclassmethod
    def reset_scene(self):
        pass

    def restrictTargetBound(self, restriction_type: str):
        self.target.set_restrictedBoundaries(restriction_type)