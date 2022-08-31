import argparse
import time
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





parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', '--number_demos', metavar="", type=int,
                    default=None,
                    help='Number of demonstrations')
parser.add_argument('-sh', '--shape', metavar="", type=str,
                    default=False,
                    help='Shape to test')
parser.add_argument('-t', '--task', metavar="", type=str,
                    default="distr",
                    help='Which models to test')

parsed_args = parser.parse_args()
n_demos = parsed_args.number_demos
shape = parsed_args.shape
task = parsed_args.task


scene_type = "shape"
saved_locations = f"{shape}_envs" #"LinearGrasp"
config_file_name = "thesis_models" #"cnn2fc_config" "model_config"
restriction_type = "same"
use_metrics = True
max_n_steps = 140
show_testing = False
n_distractors = 1

stop_between_models = False


################################################# Model
model_filename = f"BaselineCNN_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"SpatialAE_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net_deeperAttention_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net_auxiliary_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net_indepAE_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net64_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net_delta1_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"MI_Net_unfiltered_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"Future_Net_{task}_{n_demos}d"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")


