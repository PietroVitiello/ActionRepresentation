from os.path import dirname, join, abspath
from matplotlib import cm
from PIL import Image
import numpy as np
import pandas as pd
import os

# from pyrep import PyRep
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

from data_generator import Generator

SCENE_FILE = "Simulations/coppelia_robot_arm.ttt"
SAVING_DIR = join(dirname(abspath(__file__)), "Dataset/try_1")

gen = Generator(SCENE_FILE, 32)

n_demos = 100 #total number of demonstrations
n_trj = 2 #number of different trajectories for each target position
n_targetPos = np.ceil(n_demos/n_trj)
n_demos = n_targetPos * n_trj

n_episodes = 2 #50
n_runs = 2
n_steps = 100

move_type = 'linear'
constraint = 'normal'
desired_time = n_steps * 0.05

#dataframe
col_name = ["imLoc","jVel","jPos","eeVel","eePos","cPos"]
df = pd.DataFrame(columns = col_name)

for ep in range(n_episodes):
    for r in range(n_runs):
        s = 0
        if r == 0:
            move_type, constraint = 'linear', 'constrained'
            time = desired_time
        else:
            move_type, constraint = 'curved', 'constrained'
            time = desired_time * 0.95

        for data in gen.getGenerator(move_type, constraint, time):
            location = f"/images/episode_{ep}/run_{r}"
            # if not os.path.exists(SAVING_DIR + location):
            os.makedirs(SAVING_DIR + location, exist_ok=True)
            location += f"/step_{s}.jpg"

            im, joint_vel, joint_pos, ee_vel, ee_pos, cube_pos = data
            joint_vel = ",".join(np.array(joint_vel).astype(str))
            joint_pos = ",".join(np.array(joint_pos).astype(str))
            ee_pos = ",".join(ee_pos.astype(str))
            ee_vel = ",".join(ee_vel.astype(str))
            cube_pos = ",".join(cube_pos.astype(str))

            im = Image.fromarray(np.uint8(im*255)).convert('RGB') #cm.gist_earth(im/255)*255)
            # im = Image.fromarray(im)
            im.save(SAVING_DIR + location)
            # im.save("try.jpg")

            row = [location, joint_vel, joint_pos, ee_vel, ee_pos, cube_pos]
            df_length = len(df)
            df.loc[df_length] = row
            s += 1
        
        gen.resetRun()
    gen.resetEpisode()

# print(df)
# print(df[:]["eeVel"].iloc[3])
df.to_csv(SAVING_DIR + "/" + "data.csv")


gen.terminate()
