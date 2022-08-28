import argparse
from ruamel.yaml import YAML
from Learning.test_model import model_testing
from Learning.Testing.Get_Locations.gen_test_locations import gather_valid_test_positions

ryaml = YAML()

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

def get_valid_test_positions():
    print(f"Gathering {num_episodes} valid cube positions")
    gather_valid_test_positions(
        scene_type=scene_type,
        file_name=saved_locations,
        boundary_restiction="moderate",
        n_episodes=num_episodes,
        n_steps=100,
        bot_type="Baxter",
        max_deviation=0.03,
        always_maxDev=True,
        trj_type="graspDemo",
        distance_cubeReached=0.02,
        n_distractors=n_distractors,
        shape=shape
    )



scene_type = "cube"
n_distractors = 3
shape = 'tall cylinder'

model_filename = "spacer_cube"
restriction_type = "same"

saved_locations = "cube_envs" #"LinearGrasp"
use_metrics = True

num_episodes = 15
max_n_steps = 140

config_file_name = "thesis_models" #"cnn2fc_config" "model_config"

show_testing = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-gtp', '--get_test_positions', metavar="", type=bool,
                        default=False,
                        help='A boolean flag on whether to test a model (False) or gather valid cube test positions (True)')

    get_test_positions = parser.parse_args().get_test_positions
    if get_test_positions is False:
        runTest()
    else:
        get_valid_test_positions()
