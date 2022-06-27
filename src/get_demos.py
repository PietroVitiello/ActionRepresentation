import yaml
from Demos.get_data import generate_dataset

cubeReaching_generators = [""]

def runConfig():
    # if trj_type in 
    configs = {}
    configs["bot_type"] = bot_type
    configs["trj_type"] = trj_type
    configs["n_episodes"] = n_episodes
    configs["n_runs"] = n_runs
    configs["n_steps"] = n_steps
    configs["constrained"] = True #change
    
    print(f"Generating {n_episodes*n_runs} Demonstrations \n")
    d2c = generate_dataset(
            file_name,
            n_episodes,
            n_runs,
            n_steps,
            bot_type,
            trj_type,
            distance_cubeReached
        )
    print("\nDone")

    configs["distance_cubeReached"] = d2c
    print("Uploading configuration details")
    saveConfig(configs)
    print("Configurations saved in 'Demos/Dataset/descriptions.yaml'")

def saveConfig(configs):
    with open("Demos/Dataset/descriptions.yaml", 'r+') as file:
        yaml.safe_load(file)
        file.write("\n")
        dataset = {f"{file_name}": configs}
        yaml.dump(dataset, file, sort_keys=False)

file_name = "followDummy_3"
n_episodes = 100
n_runs = 1
n_steps = 100
bot_type = "Baxter"
trj_type = "LinearTrj"
distance_cubeReached = 0.01

runConfig()