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
    configs["max_deviation"] = max_deviation
    configs["always_maxDev"] = always_maxDev
    configs["boundary_restriction"] = boundary_restriction
    
    print(f"Generating {n_episodes*n_runs} Demonstrations \n")
    changed_data = generate_dataset(
                    file_name,
                    boundary_restriction,
                    n_episodes,
                    n_runs,
                    n_steps,
                    bot_type,
                    max_deviation,
                    always_maxDev,
                    trj_type,
                    distance_cubeReached
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

file_name = "HumanLikeDemos_2"
trj_type = "HumanTrj"
distance_cubeReached = 0.02
boundary_restriction = "moderate"

n_episodes = 100
n_runs = 1
n_steps = 100

bot_type = "Baxter"
max_deviation = 0.03
always_maxDev = True

runConfig()