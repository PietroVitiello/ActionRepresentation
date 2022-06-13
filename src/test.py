from os.path import dirname, join, abspath
import numpy as np
import torch
from torchvision import transforms

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor

from Demos.my_robot import MyRobot
from Demos.target import Target
from Demos.robot_movement import RobotMovement
from Learning.b_cloning.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN

pr = PyRep()

SCENE_FILE = join(dirname(abspath(__file__)), "Demos/Simulations/coppelia_robot_arm.ttt")
pr.launch(SCENE_FILE, headless=False)
pr.start()
pr.step_ui()

# arm = get_arm()
bot = MyRobot()
target = Target()
camera = VisionSensor("Vision_sensor")
rmove = RobotMovement(bot, target, pr, camera, res=64)
target.set_restrictedBoundaries()


device = torch.device('cpu')

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
#need to transform and need to normalize after
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ]
    )

model = LSTM_largerBaseCNN(9, 3)
model.load_state_dict(torch.load("Learning/b_cloning/TrainedModels/LSTM_largerBaseCNN_follow_2.pt"))
model.eval()

num_testEpisodes = 32
num_reached = 0
max_n_steps = 140

for episode in range(num_testEpisodes):
    print(f"Beginning episode {episode+1}")
    model.start_newSeq()
    target.random_pos()
    bot.resetInitial(pr)
    rmove.stayStill(2)
    reached = False
    step_n = 0
    while reached == False and step_n<max_n_steps:
        rmove.autonomousMovement(model, transform)
        reached = rmove.check_cubeReached(0.04)
        step_n += 1

    if reached:
        print("Cube reached!\n")
        num_reached += 1
    else:
        print("Cube not reached\n")

print(f"The robot was able to reach {num_reached} targets out of {num_testEpisodes}")



pr.stop()
pr.shutdown()