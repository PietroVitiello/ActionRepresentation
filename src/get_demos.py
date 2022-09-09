import yaml
import numpy as np
from Demos.get_data import generate_dataset

cubeReaching_generators = [""]

def runConfig():
    # if trj_type in 
    configs = {}
    configs["bot_type"] = bot_type
    configs["trj_type"] = trj_type
    configs["scene_type"] = scene_type
    configs["n_episodes"] = n_episodes
    if add_validation:
        n_val_episodes = int(np.round(n_episodes / 5))
        configs["n_val_episodes"] = n_val_episodes
    else:
        configs["n_val_episodes"] = 0
    configs["n_runs"] = n_runs
    configs["n_steps"] = n_steps
    configs["max_deviation"] = max_deviation
    configs["always_maxDev"] = always_maxDev
    configs["boundary_restriction"] = boundary_restriction
    configs["image_size"] = image_size
    
    print(f"********************** {scene_type} **********************")
    print(f"Generating {(n_episodes + n_val_episodes)*n_runs} Demonstrations \n")
    changed_data = generate_dataset(
                    file_name,
                    boundary_restriction,
                    n_episodes + n_val_episodes,
                    n_runs,
                    n_steps,
                    bot_type,
                    max_deviation,
                    always_maxDev,
                    trj_type,
                    distance_cubeReached,
                    image_size,
                    scene_type
                )
    print("\nDone")

    configs["distance_cubeReached"] = changed_data[0]
    configs["constrained"] = changed_data[1]

    print("Uploading configuration details")
    saveConfig(configs)
    print("Configurations saved in 'Demos/Dataset/descriptions.yaml'")

def saveConfig(configs):
    with open("Demos/Dataset/descriptions.yaml", 'r+') as file:
        yaml.safe_load(file)
        file.write("\n")
        dataset = {f"{file_name}": configs}
        yaml.dump(dataset, file, sort_keys=False)

file_name = "cubeGrasp_vis_64"
trj_type = "graspDemo"
scene_type = "cube"
distance_cubeReached = 0.02
boundary_restriction = "visible"

n_episodes = 100
n_runs = 1
n_steps = 100
add_validation = True

bot_type = "Baxter"
max_deviation = 0.03
always_maxDev = True

image_size = 64

runConfig()