from os.path import dirname, join, abspath
import numpy as np
import torch
import torchvision.transforms as T

from pyrep import PyRep

from .utils.utils_pipeline import model_choice, getModelData, testMethod
from .testing import Test

def model_testing(
    model_filename,
    num_episodes = 32,
    max_n_steps = 140
):
    pr = PyRep()

    SCENE_FILE = join(dirname(abspath(__file__)), "../Demos/Simulations/baxter_robot_arm.ttt")
    pr.launch(SCENE_FILE, headless=False)
    pr.start()
    pr.step_ui()

    # device = torch.device('cpu')

    model_name, constrained, model_params = getModelData(model_filename)
    model = model_choice(model_name, *model_params)
    model.load_state_dict(torch.load(f"Learning/TrainedModels/{model_filename}.pt"))
    model.eval()

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    #need to transform and need to normalize after
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean.tolist(), std.tolist())
        ]
    )

    test = Test(pr, model, transform, camera_res=64, num_episodes=num_episodes, max_n_steps=max_n_steps)
    reached = testMethod(test, model_name, constrained)

    pr.stop()
    pr.shutdown()

    return reached