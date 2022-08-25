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

from Demos.Scenes.distr_scene import Distractor_Scene

def gather_distr_test_positions(
    file_name: str,
    boundary_restiction: str,
    n_episodes: int,
    n_steps: int,
    bot_type: str,
    max_deviation: float,
    always_maxDev: float,
    trj_type: str,
    distance_cubeReached: float,
    n_distractors: int
):

    SCENE_FILE = join(dirname(abspath(__file__)), "../../../Demos/Simulations/baxter_robot_arm.ttt")
    SAVING_DIR = join(dirname(abspath(__file__)), "../", file_name)
    os.makedirs(SAVING_DIR, exist_ok=True)

    pr = PyRep()
    pr.launch(SCENE_FILE, headless=True)
    pr.start()
    pr.step_ui()

    bot = choseBot(bot_type)
    scene = Distractor_Scene(n_distractors)
    gen = DataGenerator(pr, scene, bot, 32, max_deviation, always_maxDev)
    gen.restrictTargetBound(boundary_restiction)

    desired_time = n_steps * 0.05
    distance_cubeReached, constrained, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)

    #dataframe
    df_columns = get_df_columns_names(n_distractors)
    pos_data = np.zeros((1, len(df_columns)))

    ep = 0
    while ep < n_episodes:
        # print(ep)
        print(f"\nTesting location {ep+1}", end="         ")
        _, _, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)
        cube_position = gen.get_cube_pos()
        distr_positions = scene.get_distractors_state(as_concatenated_array=True)

        for data in gen_process:
            task_completed = data[0]
        
        if task_completed:
            positions = np.hstack((cube_position, distr_positions))
            pos_data = np.vstack((pos_data, np.expand_dims(positions, axis=0)))
        else:
            ep -= 1
            print(f"\033[31mInvalid demo, generating new scene\033[38;5;231m", end="")
        ep += 1
        gen.resetEpisode()

    print("") #Just so that the following prints are in next line

    scene_pos_df = pd.DataFrame(data=pos_data[1:], columns=df_columns)
    scene_pos_df.to_csv(SAVING_DIR + "/" + f"valid_positions.csv")
    gen.terminate()



def get_df_columns_names(n_distractors):
    columns=['cube_x', 'cube_y', 'cube_z']
    for i in range(n_distractors):
        distr_col = [f"distr_{i+1}_x", f"distr_{i+1}_y", f"distr_{i+1}_z", f"distr_{i+1}_angle"]
        columns = columns + distr_col
    return columns

def choseBot(bot_name: str) -> MyRobot:
    if bot_name=="Mico":
        return MicoBot()
    elif bot_name=="Baxter":
        return BaxterBot()
    else:
        raise Exception("The chosen robot is not available")

def choseTrjGenrator(gen: DataGenerator, trj_type: str, time: float, distance2cube: float) -> Generator:
    if trj_type=="graspDemo":
        return *gen.getAntiCollisionGraspGenerator(time),
    else:
        raise Exception("The chosen generation process does not exist")