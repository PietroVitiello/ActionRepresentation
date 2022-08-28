from abc import abstractclassmethod
from Robotics.target import Target

class Scene():
    def __init__(
        self,
    ) -> None:
        self.target: Target = None

    @abstractclassmethod
    def set_target_object(self, target):
        self.target = target
        self._environment_startup()

    @abstractclassmethod
    def _environment_startup(self):
        pass

    @abstractclassmethod
    def reset_scene(self):
        pass

    @abstractclassmethod
    def set_scene(self, data):
        pass

    @abstractclassmethod
    def log_test_run(self, run, n_completions):
        pass

    def restrictTargetBound(self, restriction_type: str):
        self.target.set_restrictedBoundaries(restriction_type)