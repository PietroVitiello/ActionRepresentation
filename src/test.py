from ruamel.yaml import YAML
from Learning.test_model import model_testing

ryaml = YAML()

def runTest():
    cube_reached = model_testing(
        model_filename,
        num_episodes,
        max_n_steps
    )
    editConfig(cube_reached)

def editConfig(cube_reached):
    with open("Learning/TrainedModels/model_config.yaml", 'r') as file:
        configs = ryaml.load(file)
    with open("Learning/TrainedModels/model_config.yaml", 'w') as file:
        configs[model_filename]["Testing"]["Cube_Reached"].append(cube_reached)
        configs[model_filename]["Testing"]["Attempts"] = num_episodes
        ryaml.dump(configs, file)




model_filename = "Aux_BaselineCNN_3"

num_episodes = 32
max_n_steps = 140

runTest()
