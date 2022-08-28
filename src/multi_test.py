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


scene_type = "cube"
saved_locations = "cube_envs" #"LinearGrasp"
config_file_name = "thesis_models" #"cnn2fc_config" "model_config"
restriction_type = "same"
use_metrics = True
max_n_steps = 140
n_distractors = 1
show_testing = False

stop_between_models = False




n_demos = [30, 70]
n_stop_epochs = [30, 40, 50]
for n in n_demos:

    for epoch in n_stop_epochs:
        ################################################# Model
        model_filename = f"MI_Net_cube_{n}d_{epoch}es"
        runTest()
        if stop_between_models:
            input("\n\nPress enter to test the next model\n\n")

################################################# Model
model_filename = f"baseline_try"
runTest()
if stop_between_models:
    input("\n\nPress enter to test the next model\n\n")






























# ################################################# Model
# model_filename = "MI_Net_cube_30d_7e4lr"
# runTest()
# if stop_between_models:
#     input("\n\nPress enter to test the next model\n\n")

# ################################################# Model
# model_filename = "MI_Net_cube_30d_8e5lr"
# runTest()
# if stop_between_models:
#     input("\n\nPress enter to test the next model\n\n")

# ################################################# Model
# model_filename = "MI_Net_cube_30d_1e4lr"
# runTest()
# if stop_between_models:
#     input("\n\nPress enter to test the next model\n\n")

# ################################################# Model
# model_filename = "MI_Net_cube_30d_3e3lr"
# runTest()
# if stop_between_models:
#     input("\n\nPress enter to test the next model\n\n")

# ################################################# Model
# model_filename = "MI_Net_cube_30d_6e3lr"
# runTest()
# if stop_between_models:
#     input("\n\nPress enter to test the next model\n\n")