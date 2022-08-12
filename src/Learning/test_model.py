from os.path import dirname, join, abspath
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from pyrep import PyRep

from .utils.utils_pipeline import model_choice, getModelData, testMethod, getRestriction
from .Testing.testing import Test

def model_testing(
    model_filename,
    num_episodes = 32,
    max_n_steps = 140,
    restriction_type = "same",
    saved_positions = None
):
    pr = PyRep()

    SCENE_FILE = join(dirname(abspath(__file__)), "../Demos/Simulations/baxter_robot_arm.ttt")
    pr.launch(SCENE_FILE, headless=False)
    pr.start()
    pr.step_ui()

    # device = torch.device('cpu')

    model_name, constrained, dataset_name, model_params = getModelData(model_filename)
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

    restriction_type = getRestriction(restriction_type, dataset_name)
    use_saved_locations = False
    if saved_positions is not None:
        saved_positions = pd.read_csv(f"Learning/TrainedModels/{saved_positions}.csv" + "data.csv")
        use_saved_locations = True

    test = Test(pr, model, transform, restriction_type, camera_res=64, num_episodes=num_episodes, max_n_steps=max_n_steps, saved_positions=saved_positions)
    reached = testMethod(test, model_name, constrained, use_saved_locations)

    pr.stop()
    pr.shutdown()

    return reached, restriction_type