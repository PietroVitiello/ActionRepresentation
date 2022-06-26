from os.path import dirname, join, abspath
import numpy as np
import torch
import torchvision.transforms as T

from pyrep import PyRep
from Learning.utils.argParser_pipeline import model_choice
from Learning.testing import Test

pr = PyRep()

SCENE_FILE = join(dirname(abspath(__file__)), "Demos/Simulations/baxter_robot_arm.ttt")
pr.launch(SCENE_FILE, headless=False)
pr.start()
pr.step_ui()

# device = torch.device('cpu')

model = model_choice("Stopping_base", 3, 9, 16)
model.load_state_dict(torch.load("Learning/TrainedModels/Stopping_base_follow_1.pt"))
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

num_episodes = 32
num_reached = 0
max_n_steps = 140

test = Test(pr, model, transform, camera_res=64, num_episodes=num_episodes, max_n_steps=max_n_steps)
test.test_eeVelGrasp()

pr.stop()
pr.shutdown()