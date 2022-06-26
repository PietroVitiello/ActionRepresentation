from os.path import dirname, join, abspath
from typing import Callable, Generator
from PIL import Image
import numpy as np
import pandas as pd
import os

from pyrep import PyRep
# from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
# from pyrep.robots.arms.panda import Panda
# from pyrep.robots.end_effectors.mico_gripper import MicoGripper
# from pyrep.const import ObjectType, PrimitiveShape
# from pyrep.objects.vision_sensor import VisionSensor
# from pyrep.objects.shape import Shape
# from quadratic import Quadratic

# from my_robot import MyRobot
# from target import Target

# import time
# import math

from Demos.data_generator import DataGenerator
from Robotics.Robot.micoBot import MicoBot
from Robotics.Robot.baxterBot import BaxterBot
from Robotics.Robot.my_robot import MyRobot

def generate_dataset(
    file_name,
    n_episodes,
    n_runs,
    n_steps,
    bot_type,
    trj_type,
    distance_cubeReached
):

    SCENE_FILE = join(dirname(abspath(__file__)), "Simulations/baxter_robot_arm.ttt")
    SAVING_DIR = join(dirname(abspath(__file__)), f"Dataset/{file_name}") #followDummy_fixed_2

    pr = PyRep()
    pr.launch(SCENE_FILE, headless=True)
    pr.start()
    pr.step_ui()

    bot = choseBot(bot_type)
    gen = DataGenerator(pr, bot, 64)
    gen.restrictTargetBound()

    # n_demos = 100 #total number of demonstrations
    # n_trj = 2 #number of different trajectories for each target position
    # n_targetPos = np.ceil(n_demos/n_trj)
    # n_demos = n_targetPos * n_trj

    # n_episodes = 100
    # n_runs = 1
    # n_steps = 95

    desired_time = n_steps * 0.05
    distance_cubeReached, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)

    #dataframe
    col_name = ["imLoc","j_targetVel","jVel","jPos","ee_targetVel","eeVel","eePos","eeOri","cPos","stop"]
    df = pd.DataFrame(columns = col_name)

    for ep in range(n_episodes):
        for r in range(n_runs):
            print(f"Episode: {ep+1}\t Run: {r+1}")
            distance_cubeReached, gen_process = choseTrjGenrator(gen, trj_type, desired_time, distance_cubeReached)
            s = 0
            for data in gen_process:
                location = f"/images/episode_{ep}/run_{r}"
                # if not os.path.exists(SAVING_DIR + location):
                os.makedirs(SAVING_DIR + location, exist_ok=True)
                location += f"/step_{s}.jpg"
                
                ee_target, im, joint_target, joint_vel, joint_pos, ee_vel, ee_pos, ee_orientation, cube_pos, stop = data
                joint_target = ",".join(np.array(joint_target).astype(str))
                joint_vel = ",".join(np.array(joint_vel).astype(str))
                joint_pos = ",".join(np.array(joint_pos).astype(str))
                ee_target = ",".join(ee_target.astype(str))
                ee_vel = ",".join(ee_vel.astype(str))
                ee_pos = ",".join(ee_pos.astype(str))
                ee_orientation = ",".join(ee_orientation.astype(str))
                cube_pos = ",".join(cube_pos.astype(str))

                im = Image.fromarray(np.uint8(im*255)).convert('RGB') #cm.gist_earth(im/255)*255)
                # im = Image.fromarray(im)
                im.save(SAVING_DIR + location)
                # im.save("try.jpg")

                row = [location, joint_target, joint_vel, joint_pos, ee_target, ee_vel, ee_pos, ee_orientation, cube_pos, stop]
                df_length = len(df)
                df.loc[df_length] = row
                s += 1
            
            gen.resetRun()
        gen.resetEpisode()

    df.to_csv(SAVING_DIR + "/" + "data.csv")

    gen.terminate()

    return distance_cubeReached

def choseBot(bot_name: str) -> MyRobot:
    if bot_name=="Mico":
        return MicoBot()
    elif bot_name=="Baxter":
        return BaxterBot()
    else:
        raise Exception("The chosen robot is not available")


def choseTrjGenrator(gen: DataGenerator, trj_type: str, time: float, distance2cube: float) -> Generator:
    if trj_type=="HumanTrj":
        return distance2cube, gen.getHumanTrjGenerator(time, distance2cube)
    elif trj_type=="HumanTrj_fixedSteps":
        return None, gen.humanTrjGenerator_fixedSteps(time)
    elif trj_type=="HumanTrj_imperfect":
        return distance2cube, gen.getHumanTrjGenerator_imperfect (time, distance2cube)
    elif trj_type=="HumanTrj_stop":
        return *gen.getHumanTrjGenerator_stop(time),
    elif trj_type=="LinearTrj":
        return distance2cube, gen.getLinearTrjGenerator(time)
    else:
        raise Exception("The chosen generation process does not exist")


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
