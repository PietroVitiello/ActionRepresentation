# A New Representation for Actions in Visual Robot Learning
This repo was created for a project aimed at finding a new representation for robot actions. The work environment consists in a robotic arm with a camera mounted on its wrist. An agent is then trained through imitation learning to control this robotic arm by receiving as input the images captured by the camera. The presented work has been fully developed in simulation and here you will find:
* code to interface with a simulated environment and control it
* a kinematics package to control a robotic arm to reach and grasp a target object
* code to generate different simulation scenes
* code to create demonstrations of the robotic arm grasping objects in these scenes
* various neural networks that process an image and return a robot action
* code to train these models using different data pipelines
* code to test the models by using them to control the robot in simulation

Table of contents
=================

<!--ts-->
   * [Project Motivation](#project-motivation)
   * [Requirements](#requirements)
   * [Code Structure](#code-structure)
   * [Scenes](#scenes)
      * [Red Cube](#red-cube)
      * [Red Cube with Distractor Objects](#red-cube-with-distractor-objects)
      * [Grasping Different Shapes](#grasping-different-shapes)
   * [Proposed Novel Action Representation](#proposed-novel-action-representation)
      * [Motion Image](#the-motion-image)
      * [MI-Net](#mi-net)
<!--te-->

Project Motivation
=================
The field of robotics has been gaining a lot of traction, especially in the medical, military and industrial sectors. However, the majority of the currently used robots rely on human-crafted control systems, which take time to be programmed and have a low tolerance for variations in their environment. Consequently, Robot Learning tries to use artificial intelligence to teach robots how to autonomously perform tasks, with the ultimate goal of creating general agents that could perform in any environment without having to be reprogrammed. Usually these agents take sensory input, such as images or cloud-point data (like LiDAR), and process it in order to come up with instructions to give to the robot.

These instructions are effectively robot actions, or in other words movements that the robot should perform. Nonetheless, a question arises on how one should represent these actions. A robot action could for example be represented as a vector containing the linear and angular velocities of the end-effector, or as a location and orientation in space that the end-effector should reach. Representation Learning teaches us that how data is represented in a machine learning model has a great impact on its performance. Therefore an agent could more easily predict actions when they are represented in a form compared to another. This project tries to explore a new way of representing robot actions, not as vectors but as images. The reason being that neural networks might find it easier to process images rather than vectors and a visual representation could be also more easily interpretable by a human being.

Requirements
=================
This project has been fully developed in simulated environments. The simulator used was [CoppeliaSim V4.1](https://www.coppeliarobotics.com/downloads) which was controlled via python scripts through the use of the [PyRep](https://github.com/stepjam/PyRep) python library. In order for the latter to work one must run the scripts in the Linux operating system.

Apart from these major requirements, this repo relies on a few python libraries that can be simply pip installed by running ``` pip install -r requirements.txt ```

Code Structure
=================
The main code of this project resides in the ```src``` folder. More precisely, each folder has the following functionality 
* <ins>__/Demos__</ins>: This folder has the code used to interact with the simulations. All of the actual simulation files are contained in this folder. Moreover, the code necessary to create the experiment scenes and generate demonstrations is also found here.

* <ins>__/Robotics__</ins>: This folder hosts the code that actually controls the robot. Here you can find the classes that interface and communicate with the robot in the simulated scene. Additionally the full kinematic controls of the robot can be found here.

* <ins>__/Learning__</ins>: This folder contains all of the models that have been implemented. Some of them use autoencoders, others LSTMs, other only convolutional layers etc. In this folder you can also find all the custom dataloaders as well as various methods to train and test the models.

These folders contain all of the source code. However the main scripts of the repo are the following and they can be used to control the entire project:
* <ins>__get_demos.py__</ins>: In this script you can chose a simulation environment as well as set some demonstration parameters. By running this script you will generate demonstrations, automatically saving the demonstration data in the dataset folder ```src/Demos/Dataset``` as well as saving the configurations used to create such dataset into the ```descriptions.yaml``` file.

* <ins>__train.py__</ins>: In this script you can chose a model and its configuration as well as a dataset from does stored in the ```src/Demos/Dataset``` folder. By running this script the chosen model will be trained on the chosen dataset. The trained model will be saved in the ```src/Learning/TrainedModels``` folder and in this same folder the configurations of the model will be saved in a yaml file.

* <ins>__test.py__</ins>: In this script you can chose the trained model to test and the scene in which to test it. By running this script the model will be tested on grasping objects and the number of successful grasps will be automatically saved.

 

Scenes
=================
There are three main scenes that have been used to train and test the models. The source code to define and generate them can be found in ```src/Demos/Scenes```, while the code to collect demonstrations using these scenes can be found in ```src/Demos/DemoCollection```

## Red Cube
Here a red cube was randomly initialised in front of the robotic arm. The objective is to successfully grasp the cube. Below you can find images of an example of such a scene both in third (left) and first (right) person views.
<p float="center">
  <img src="/Images/cube_env.png" height="300">
  <img src="/Images/cube_scene_2.jpg" height="300">
</p>

## Red Cube with Distractor Objects
Here the objective remains the same as above. However, in the scene also distractor objects get randomly initialised. This environment can be used to test whether models can adapt and generalise to scenes which are visually different and if the models are able to still recognise that the target is the red cube although there are other objects present as well. Below you can find images of an example of such a scene both in third (left) and first (right) person views.
<p float="center">
  <img src="/Images/distr_env_2.png" height="300">
  <img src="/Images/distr_scene_2.jpg" height="300">
</p>

## Grasping Different Shapes
In this case the target object is not necessarily a cube anymore. The object gets generated with a random shape between cube, horizontal prism, vertical prism, cylinder or a taller cylinder. These shapes have different properties and need to be grasped in a slightly different way in order for the grasp to be robust enough to lift the object. As a result using this environment can test the adaptability of the models to different object shapes.

Proposed Novel Action Representation
=================
We propose an alterantive action representation that tries to represent motion in the form of an image.

## The Motion Image
This work introduces the idea of using motion images as action representation. These are generated by subtracting following frames of a video. More specifically, during demonstrations one is effectively collecting a video of what the robot camera should be seeing while performing a certain task. By subtracting the video frame at time-step t from the video frame at time step t+∆t what you obtain is an image representing the change in point of view of the camera. This image is known as the motion image and it depends on the movement that the camera has performed between time-step t and t+∆t. As a result, one can use the motion image to represent actions. Examples of motion images can be found below.

<img src="/Images/MI_45.png" height="300">

<img src="/Images/MI_95.png" height="300">

## MI-Net
To leverage the use of the motion image we propose the MI-Net which makes use of an autoencoder and attention mechanism. A decoder is trained to generate the motion image corresponding to the desired action that should be executed by the robot. The attention mechanism uses the information of the decoder to provide attention on the prediction of end-effector velocities. The code for the MI-Net and some of its alterations can be found in ```src/Learning/Models/MotionIMG```
