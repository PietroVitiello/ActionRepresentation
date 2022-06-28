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
        configs["StrengthSpatialAE_fc_follow_4"]["Testing"]["Cube_Reached"].append(cube_reached)
        configs["StrengthSpatialAE_fc_follow_4"]["Testing"]["Attempts"] = num_episodes
        ryaml.dump(configs, file)




model_filename = "StrengthSpatialAE_fc_follow_2"

num_episodes = 32
max_n_steps = 140

runTest()
