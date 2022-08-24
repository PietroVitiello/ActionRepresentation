from os.path import dirname, join, abspath
import shutil
from typing import Callable, Generator
from PIL import Image
import numpy as np
import pandas as pd
import os

from pyrep import PyRep

from Demos.data_generator import DataGenerator
from Robotics.Robot.micoBot import MicoBot
from Robotics.Robot.baxterBot import BaxterBot
from Robotics.Robot.my_robot import MyRobot

from .Scenes.distr_scene import Distractor_Scene

from .DemoCollection.cube_collect import generate_cube_dataset
from .DemoCollection.distr_collect import generate_distractor_dataset

def generate_dataset(
    file_name: str,
    boundary_restiction: str,
    n_episodes: int,
    n_runs: int,
    n_steps: int,
    bot_type: str,
    max_deviation: float,
    always_maxDev: float,
    trj_type: str,
    distance_cubeReached: float,
    image_size: int = 64,
    scene_type: str = 'cube'
):

    if scene_type == 'cube':
        generate_cube_dataset(
            file_name,
            boundary_restiction,
            n_episodes,
            n_runs,
            n_steps,
            bot_type,
            max_deviation,
            always_maxDev,
            trj_type,
            distance_cubeReached,
            image_size
        )

    elif scene_type == 'distractor':
        generate_distractor_dataset(
            file_name,
            boundary_restiction,
            n_episodes,
            n_runs,
            n_steps,
            bot_type,
            max_deviation,
            always_maxDev,
            trj_type,
            distance_cubeReached,
            image_size
        )



if __name__ == "__main__":
    print("This script needs to be ran from src/get_demos.py")























# move_type = 'linear'
# constraint = 'normal'

# for ep in range(n_episodes):
#     for r in range(n_runs):
#         s = 0
#         if r == 0:
#             move_type, constraint = 'linear', 'constrained'
#             time = desired_time
#         else:
#             move_type, constraint = 'curved', 'constrained'
#             time = desired_time * 0.95

#         for data in gen.getGenerator(move_type, constraint, time):
#             location = f"/images/episode_{ep}/run_{r}"
#             # if not os.path.exists(SAVING_DIR + location):
#             os.makedirs(SAVING_DIR + location, exist_ok=True)
#             location += f"/step_{s}.jpg"

#             im, joint_vel, joint_pos, ee_vel, ee_pos, cube_pos = data
#             joint_vel = ",".join(np.array(joint_vel).astype(str))
#             joint_pos = ",".join(np.array(joint_pos).astype(str))
#             ee_pos = ",".join(ee_pos.astype(str))
#             ee_vel = ",".join(ee_vel.astype(str))
#             cube_pos = ",".join(cube_pos.astype(str))

#             im = Image.fromarray(np.uint8(im*255)).convert('RGB') #cm.gist_earth(im/255)*255)
#             # im = Image.fromarray(im)
#             im.save(SAVING_DIR + location)
#             # im.save("try.jpg")

#             row = [location, joint_vel, joint_pos, ee_vel, ee_pos, cube_pos]
#             df_length = len(df)
#             df.loc[df_length] = row
#             s += 1
        
#         gen.resetRun()
#     gen.resetEpisode()
