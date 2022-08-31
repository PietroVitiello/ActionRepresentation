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

from Demos.Scenes.cube_scene import Cube_Scene

def gather_cube_test_positions(
    file_name: str,
    boundary_restriction: str,
    n_episodes: int,
    n_steps: int,
    bot_type: str,
    max_deviation: float,
    always_maxDev: float,
    trj_type: str,
    distance_cubeReached: float,
):

    SCENE_FILE = join(dirname(abspath(__file__)), "../../../Demos/Simulations/baxter_robot_arm.ttt")
    SAVING_DIR = join(dirname(abspath(__file__)), "../", file_name)
    os.makedirs(SAVING_DIR, exist_ok=True)

    pr = PyRep()
    pr.launch(SCENE_FILE, headless=True)
    pr.start()
    pr.step_ui()

    bot = choseBot(bot_type)
    scene = Cube_Scene()
    gen = DataGenerator(pr, scene, bot, 32, max_deviation, always_maxDev)
    gen.restrictTargetBound(boundary_restriction)

    desired_time = n_steps * 0.05
    distance_cubeReached, constrained, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)

    #dataframe
    cube_positions = np.zeros((1, 3))

    ep = 0
    while ep < n_episodes:
        # print(ep)
        print(f"\nTesting location {ep+1}", end="         ")
        _, _, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)
        cube_position = gen.get_cube_pos()

        for data in gen_process:
            task_completed = data[0]
        
        if task_completed:
            cube_positions = np.vstack((cube_positions, np.expand_dims(cube_position, axis=0)))
        else:
            ep -= 1
            print(f"\033[31mCube position [{gen.target.get_position()[:2]}] is likely invalid, generating new cube\033[38;5;231m", end="")
        ep += 1
        gen.resetEpisode()

    print("") #Just so that the following prints are in next line

    cube_pos_df = pd.DataFrame(columns=['cube_x', 'cube_y', 'cube_z'])
    cube_pos_df['cube_x'], cube_pos_df['cube_y'], cube_pos_df['cube_z'] = cube_positions[1:,:].T
    cube_pos_df.to_csv(SAVING_DIR + "/" + "valid_positions.csv")
    gen.terminate()


def choseBot(bot_name: str) -> MyRobot:
    if bot_name=="Mico":
        return MicoBot()
    elif bot_name=="Baxter":
        return BaxterBot()
    else:
        raise Exception("The chosen robot is not available")


def choseTrjGenrator(gen: DataGenerator, trj_type: str, time: float, distance2cube: float) -> Generator:
    if trj_type=="HumanTrj":
        return distance2cube, *gen.getHumanTrjGenerator(time, distance2cube)
    if trj_type=="HumanGrasp":
        return *gen.getHumanTrjGraspGenerator(time),
    elif trj_type=="FollowDummy":
        return distance2cube, *gen.getFollowDummyGenerator(time, distance2cube)
    elif trj_type=="FollowDummy_fixedSteps":
        return None, *gen.getFollowDummyGenerator_fixedSteps(time)
    elif trj_type=="HumanTrj_imperfect":
        return distance2cube, *gen.getHumanTrjGenerator_imperfect (time, distance2cube)
    elif trj_type=="FollowDummy_stop":
        return *gen.getFollowDummyGenerator_stop(time),
    elif trj_type=="LinearTrj":
        return distance2cube, *gen.getLinearTrjGenerator(time, distance2cube)
    if trj_type=="graspDemo":
        return *gen.getLinearGraspGenerator(time),
    else:
        raise Exception("The chosen generation process does not exist")