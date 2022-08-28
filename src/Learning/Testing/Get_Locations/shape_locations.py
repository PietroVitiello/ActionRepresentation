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

from Demos.Scenes.shape_scene import Shape_Scene

def gather_shape_test_positions(
    file_name: str,
    boundary_restriction: str,
    n_episodes: int,
    n_steps: int,
    bot_type: str,
    max_deviation: float,
    always_maxDev: float,
    trj_type: str,
    distance_cubeReached: float,
    shape: str = 'cube'
):

    SCENE_FILE = join(dirname(abspath(__file__)), "../../../Demos/Simulations/baxter_robot_arm.ttt")
    SAVING_DIR = join(dirname(abspath(__file__)), "../", file_name)
    os.makedirs(SAVING_DIR, exist_ok=True)

    pr = PyRep()
    pr.launch(SCENE_FILE, headless=True)
    pr.start()
    pr.step_ui()

    bot = choseBot(bot_type)
    scene = Shape_Scene(boundary_restriction, shape)
    gen = DataGenerator(pr, scene, bot, 32, max_deviation, always_maxDev)
    gen.set_target(scene.get_target())
    gen.restrictTargetBound(boundary_restriction)

    desired_time = n_steps * 0.05
    distance_cubeReached, constrained, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)

    #dataframe
    shape_index = scene.get_supported_shapes().index(shape)
    df_columns = ['target_x', 'target_y', 'target_z', 'shape']
    pos_data = np.zeros((1, len(df_columns[:-1])))

    ep = 0
    while ep < n_episodes:
        # print(ep)
        print(f"\nTesting location {ep+1}", end="         ")
        _, _, gen_process = choseTrjGenrator(gen, trj_type, desired_time, scene.get_distance2cube())
        cube_position = gen.get_cube_pos()

        for data in gen_process:
            task_completed = data[0]
        
        if task_completed:
            pos_data = np.vstack((pos_data, np.expand_dims(cube_position, axis=0)))
        else:
            ep -= 1
            print(f"\033[31mInvalid demo, generating new scene\033[38;5;231m", end="")
        ep += 1
        gen.resetEpisode()

    print("") #Just so that the following prints are in next line

    pos_data = np.hstack((pos_data, np.ones((pos_data.shape[0], 1))*shape_index))
    scene_pos_df = pd.DataFrame(data=pos_data[1:], columns=df_columns)
    scene_pos_df.to_csv(SAVING_DIR + "/" + f"valid_positions.csv")
    gen.terminate()


def choseBot(bot_name: str) -> MyRobot:
    if bot_name=="Mico":
        return MicoBot()
    elif bot_name=="Baxter":
        return BaxterBot()
    else:
        raise Exception("The chosen robot is not available")

def choseTrjGenrator(gen: DataGenerator, trj_type: str, time: float, distance2cube: float) -> Generator:
    if trj_type=="graspDemo":
        return *gen.getLinearGraspGenerator(time, distance2cube),
    else:
        raise Exception("The chosen generation process does not exist")