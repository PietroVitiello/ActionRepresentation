import yaml
from Demos.get_data import generate_dataset

cubeReaching_generators = [""]

def useConfig():
    # if trj_type in 
    saveConfig()

    generate_dataset(
        file_name,
        n_episodes,
        n_runs,
        n_steps,
        trj_type,
    )

def saveConfig():
    with open("Demos/Dataset/descriptions.yaml", 'r+') as file:
        configs = {}
        configs["n_episodes"] = n_episodes
        configs["n_runs"] = n_runs
        configs["n_steps"] = n_steps
        configs["bot_type"] = bot_type
        configs["trj_type"] = trj_type

        dataset = {f"{file_name}": configs}
        yaml.dump(dataset, file)
        file.write("\n")

file_name = "trying_4"
n_episodes = 100
n_runs = 1
n_steps = 95
bot_type = "Baxter"
trj_type = "HumanTrj_stop"
distance_cubeReached = 0.03

useConfig()