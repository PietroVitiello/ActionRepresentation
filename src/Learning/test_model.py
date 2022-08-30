from os.path import dirname, join, abspath
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from pyrep import PyRep
import wandb

from .utils.utils_pipeline import model_choice, getModelData, testMethod, getRestriction
from .utils.utils_test import get_transform, get_run_id
from .Testing.testing import Test

from Demos.Scenes.cube_scene import Cube_Scene
from Demos.Scenes.distr_scene import Distractor_Scene
from Demos.Scenes.shape_scene import Shape_Scene

def model_testing(
    model_filename,
    scene_type,
    num_episodes = 32,
    max_n_steps = 140,
    restriction_type = "same",
    saved_positions = None,
    use_metrics: bool = False,
    config_filename = "LinearGrasp",
    n_distractors = None,
    show: bool = True
):
    pr = PyRep()

    SCENE_FILE = join(dirname(abspath(__file__)), "../Demos/Simulations/baxter_robot_arm.ttt")
    pr.launch(SCENE_FILE, headless=not show)
    pr.start()
    pr.step_ui()

    run_id = get_run_id(model_filename, config_filename)
    run = wandb.init(
        project="New-Robot-Action-Representation",
        id=run_id,
        reinit=True,
        resume="allow"
    )

    model_name, constrained, dataset_name, model_params = getModelData(model_filename, config_filename)
    model = model_choice(model_name, *model_params)
    model.load_state_dict(torch.load(f"Learning/TrainedModels/{model_filename}.pt"))
    model.eval()

    data_transforms = get_transform(use_metrics, model_filename, config_filename)

    restriction_type = getRestriction(restriction_type, dataset_name)
    use_saved_locations = False
    if saved_positions is not None:
        saved_positions = pd.read_csv(f"Learning/Testing/{saved_positions}/" + "valid_positions.csv")
        use_saved_locations = True

    scene = get_scene(scene_type, n_distractors, restriction_type)

    test = Test(pr, scene, model, data_transforms, restriction_type, camera_res=64, num_episodes=num_episodes, max_n_steps=max_n_steps, saved_positions=saved_positions)
    reached = testMethod(test, model_name, constrained, use_saved_locations)
    scene.log_test_run(run, reached)

    pr.stop()
    pr.shutdown()

    return reached, restriction_type


def get_scene(scene_type, n_distractors, boundary_restriction):
    if scene_type == 'cube':
        return Cube_Scene()

    elif scene_type == 'distractor':
        return Distractor_Scene(n_distractors)

    elif scene_type == 'shape':
        return Shape_Scene(boundary_restriction)
