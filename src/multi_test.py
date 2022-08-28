import argparse
from ruamel.yaml import YAML
from Learning.test_model import model_testing

ryaml = YAML()
num_episodes = 32

def runTest():
    cube_reached, restriction = model_testing(
        model_filename,
        scene_type,
        num_episodes,
        max_n_steps,
        restriction_type,
        saved_locations,
        use_metrics,
        config_file_name,
        n_distractors,
        show_testing,
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


scene_type = "cube"
saved_locations = "cube_envs" #"LinearGrasp"
config_file_name = "thesis_models" #"cnn2fc_config" "model_config"
restriction_type = "same"
use_metrics = True
max_n_steps = 140
n_distractors = 1
show_testing = True

stop_between_models = True

################################################# Model
model_filename = "BaselineCNN_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "SpatialAE_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "MI_Net_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "MI_Net_deeperAttention_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "MI_Net64_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "MI_Net_delta1_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "MI_Net_unfiltered_cube_100d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = "Future_Net_cube_100d"
runTest()