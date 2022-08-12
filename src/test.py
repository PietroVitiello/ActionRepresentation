import argparse
from ruamel.yaml import YAML
from Learning.test_model import model_testing
from Learning.Testing.test_locations import gather_valid_test_positions

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
        if saved_locations is None:
            configs[model_filename]["Testing"]["Attempts"] = num_episodes
        else:
            configs[model_filename]["Testing"]["Attempts"] = 100
        ryaml.dump(configs, file)

def get_valid_test_positions():
    print(f"Gathering 100 valid cube positions")
    gather_valid_test_positions(
        file_name="LinearGrasp",
        boundary_restiction="moderate",
        n_episodes=100,
        n_steps=100,
        bot_type="Baxter",
        max_deviation=0.03,
        always_maxDev=True,
        trj_type="LinearGrasp",
        distance_cubeReached=0.02
    )




model_filename = "Stop_AuxBaselineCNN_5"
restriction_type = "same"
saved_locations = "LinearGrasp"

num_episodes = 32
max_n_steps = 140


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-gtp', '--get_test_positions', metavar="", type=bool,
                        default=False,
                        help='A boolean flag on whether to test a model (False) or gather valid cube test positions (True)')

    get_test_positions = parser.parse_args()
    if get_test_positions is False:
        runTest()
    else:
        get_valid_test_positions()
