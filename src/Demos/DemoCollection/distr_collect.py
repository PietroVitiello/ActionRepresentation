from os.path import dirname, join, abspath
import shutil
from typing import Callable, Generator
from PIL import Image
import numpy as np
import pandas as pd
import os

from pyrep import PyRep

from ..data_generator import DataGenerator
from Robotics.Robot.micoBot import MicoBot
from Robotics.Robot.baxterBot import BaxterBot
from Robotics.Robot.my_robot import MyRobot

from ..Scenes.distr_scene import Distractor_Scene

def generate_distractor_dataset(
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
    image_size: int = 64
):

    SCENE_FILE = join(dirname(abspath(__file__)), "../Simulations/baxter_robot_arm.ttt")
    SAVING_DIR = join(dirname(abspath(__file__)), f"../Dataset/{file_name}")

    pr = PyRep()
    pr.launch(SCENE_FILE, headless=False)
    pr.start()
    pr.step_ui()

    bot = choseBot(bot_type)
    scene = Distractor_Scene(3)
    gen = DataGenerator(pr, scene, bot, image_size, max_deviation, always_maxDev)
    gen.restrictTargetBound(boundary_restiction)

    desired_time = n_steps * 0.05
    distance_cubeReached, constrained, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)

    #dataframe
    col_name = ["demo_id","imLoc","j_targetVel","jVel","jPos","ee_targetVel","eeVel","eePos","eeOri","cPos","stop"]
    df = pd.DataFrame(columns = col_name)
    temp_df = pd.DataFrame(columns = col_name)
    cube_positions = np.zeros((1, 3))

    ep = 0
    while ep < n_episodes:
        r = 0
        taskFail_counter = 0
        while r < n_runs:
            print(f"\nEpisode: {ep+1}\t Run: {r+1}", end="         ")
            demo_id = ep*n_runs + r
            _, _, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)
            s = 0
            for data in gen_process:
                location = f"/images/episode_{ep}/run_{r}"
                os.makedirs(SAVING_DIR + location, exist_ok=True)
                location += f"/step_{s}.jpg"
                
                task_completed = data[0]
                ee_target, im, joint_target, joint_vel, joint_pos, ee_vel, ee_pos, ee_orientation, cube_pos, stop = data[1]

                joint_target = ",".join(np.array(joint_target).astype(str))
                joint_vel = ",".join(np.array(joint_vel).astype(str))
                joint_pos = ",".join(np.array(joint_pos).astype(str))
                ee_target = ",".join(ee_target.astype(str))
                ee_vel = ",".join(ee_vel.astype(str))
                ee_pos = ",".join(ee_pos.astype(str))
                ee_orientation = ",".join(ee_orientation.astype(str))
                cube_pos = ",".join(cube_pos.astype(str))

                im = Image.fromarray(np.uint8(im*255)).convert('RGB') #cm.gist_earth(im/255)*255)
                im.save(SAVING_DIR + location)

                row = [demo_id, location, joint_target, joint_vel, joint_pos, ee_target, ee_vel, ee_pos, ee_orientation, cube_pos, stop]
                df_length = len(temp_df)
                temp_df.loc[df_length] = row
                s += 1
            
            if task_completed:
                df = pd.concat((df, temp_df), axis=0)
                cube_positions = np.vstack((cube_positions, np.expand_dims(gen.get_cube_pos(), axis=0)))
                taskFail_counter = 0
                r += 1
            else:
                taskFail_counter += 1
                # if taskFail_counter == 5:
                try:
                    shutil.rmtree(SAVING_DIR + f"/images/episode_{ep}")
                except IsADirectoryError:
                    pass
                r = n_runs
                ep -= 1
                print(f"\033[31mCube position [{gen.target.get_position()[:2]}] is likely invalid, generating new cube\033[38;5;231m", end="")
                # else:
                #     shutil.rmtree(SAVING_DIR + f"/images/episode_{ep}/run_{r}")
                #     print("\033[31mTask not completed, repeating run\033[38;5;231m", end="")

            temp_df = pd.DataFrame(columns = col_name) #empty temp_df
            gen.resetRun()
        ep += 1
        gen.resetEpisode()

    print("") #Just so that the following prints are in next line

    cube_pos_df = pd.DataFrame(columns=['cube_x', 'cube_y', 'cube_z'])
    cube_pos_df['cube_x'], cube_pos_df['cube_y'], cube_pos_df['cube_z'] = cube_positions[1:,:].T

    df.to_csv(SAVING_DIR + "/" + "data.csv")
    cube_pos_df.to_csv(SAVING_DIR + "/" + "cube_positions.csv")

    gen.terminate()

    return distance_cubeReached, constrained




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
    if trj_type=="LinearGrasp":
        return *gen.getAntiCollisionGraspGenerator(time),
    else:
        raise Exception("The chosen generation process does not exist")


if __name__ == "__main__":
    print("This script needs to be ran from src/get_demos.py")
