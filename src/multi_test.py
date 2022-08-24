import argparse
from ruamel.yaml import YAML
from Learning.test_model import model_testing

ryaml = YAML()

def runTest():
    cube_reached, restriction = model_testing(
        model_filename,
        num_episodes,
        max_n_steps,
        restriction_type,
        saved_locations,
        use_metrics,
        config_file_name,
        show_testing
    )
    editConfig(cube_reached, restriction)

def editConfig(cube_reached, restriction):
    with open(f"Learning/TrainedModels/{config_file_name}.yaml", 'r') as file:
        configs = ryaml.load(file)
    with open(f"Learning/TrainedModels/{config_file_name}.yaml", 'w') as file:
        configs[model_filename]["Testing"]["Cube_Reached"].append(cube_reached)
        configs[model_filename]["Testing"]["Boundary_Restriction"].append(restriction)
        if saved_locations is None:
            configs[model_filename]["Testing"]["Attempts"] = num_episodes
        else:
            configs[model_filename]["Testing"]["Attempts"] = 100
        ryaml.dump(configs, file)


show_testing = True

################################################# Model
model_filename = "discard_mi"
restriction_type = "same"

saved_locations = None #"LinearGrasp"
use_metrics = True

num_episodes = 32
max_n_steps = 140

config_file_name = "model_config" #"cnn2fc_config" "model_config"


runTest()