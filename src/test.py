from ruamel.yaml import YAML
from Learning.test_model import model_testing

ryaml = YAML()

def runTest():
    cube_reached, restriction = model_testing(
        model_filename,
        num_episodes,
        max_n_steps,
        restriction_type
    )
    editConfig(cube_reached, restriction)

def editConfig(cube_reached, restriction):
    with open("Learning/TrainedModels/model_config.yaml", 'r') as file:
        configs = ryaml.load(file)
    with open("Learning/TrainedModels/model_config.yaml", 'w') as file:
        configs[model_filename]["Testing"]["Cube_Reached"].append(cube_reached)
        configs[model_filename]["Testing"]["Boundary_Restriction"].append(restriction)
        configs[model_filename]["Testing"]["Attempts"] = num_episodes
        ryaml.dump(configs, file)




model_filename = "Aux_BaselineCNN_3"
restriction_type = "same"

num_episodes = 32
max_n_steps = 140

runTest()
