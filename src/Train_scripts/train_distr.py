import yaml
import numpy as np
from Learning.train_model import loading_data, training_individual

dataset_pipe = None
saved_config = "thesis_models" # "cnn2fc_config"
tags = None

def load_data():
    print(f"\n{int(np.round(n_demos * (2-train_val_split)))} Total Demonstrations: {n_demos} T, {int(np.round(n_demos * (1-train_val_split)))} V")
    transform, metrics, reach_datasets, stop_datasets, ds = loading_data(
                                                            data_folder,
                                                            [reach_data_mode, stop_data_mode],
                                                            n_demos,
                                                            train_val_split,
                                                            dataset_pipe,
                                                            is_shape_data
                                                        )
    return transform, metrics, reach_datasets, stop_datasets, ds

def runConfig(transform, metrics, reach_datasets, stop_datasets):
    configs = {}
    configs["model_name"] = model_name
    configs["data_folder"] = data_folder
    configs["training_method"] = training_method
    configs["n_epochs"] = epochs
    configs["n_epochs_stopping"] = epochs_stopping
    configs["batch_size"] = batch_size
    configs["use_gpu"] = use_gpu
    configs["n_demos_used"] = n_demos
    configs["train_val_split"] = train_val_split
    configs["optimiser"] = optimiser
    configs["learning_rate"] = lr
    configs["weight_decay"] = weight_decay
    configs["loss"] = loss
    configs["stopping_loss"] = stopping_loss
    configs["num_outputs"] = num_outputs
    configs["num_aux_outputs"] = num_aux_outputs
    configs["reconstruction_size"] = recon_size

    print("\n", "#"*30)
    print(f"Training {saved_model_name} Model\n")
    useless_keys, run_id = training_individual(
                        reach_datasets,
                        stop_datasets,
                        transform,
                        saved_model_name,
                        epochs,
                        epochs_stopping,
                        batch_size,
                        training_method,
                        use_gpu,
                        optimiser,
                        lr,
                        weight_decay,
                        loss,
                        stopping_loss,
                        model_name,
                        num_outputs,
                        num_aux_outputs,
                        recon_size,
                        tags
                    )
    keepUseful(configs, useless_keys)
    # print(metrics[0])
    configs["input_metrics"] = [metrics[0][0].tolist(), metrics[0][1].tolist()]
    if metrics[1][0] is not None:
        configs["recon_metrics"] = [metrics[1][0].tolist(), metrics[1][1].tolist()]
    configs["eeVel_metrics"] = [metrics[2][0].tolist(), metrics[2][1].tolist()]
    configs["aux_metrics"] = [metrics[3][0].tolist(), metrics[3][1].tolist()]
    configs["id"] = run_id

    print("Uploading configuration details")
    saveConfig(configs)
    print("Configurations saved in 'Learning/TrainedModels/model_config.yaml'\n")
    

def saveConfig(configs):
    with open(f"Learning/TrainedModels/{saved_config}.yaml", 'r+') as file:
        yaml.safe_load(file)
        file.write("\n")
        configs["Testing"] = {}
        configs["Testing"]["Cube_Reached"] = []
        configs["Testing"]["Boundary_Restriction"] = []
        configs["Testing"]["Attempts"] = None
        model_config = {f"{saved_model_name}": configs}
        yaml.dump(model_config, file, sort_keys=False)
        file.write("#NOTE:\n")

def keepUseful(configs:dict, useless: list):
    for key in useless:
        configs.pop(key)




#****************** Spacer Run ******************#
data_folder = "cubeGrasp_64"
is_shape_data = False
reach_data_mode = "aux"
stop_data_mode = "onlyStop"
n_demos = 1
train_val_split = 0.8
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()
epochs = 1
batch_size = 64
use_gpu = True
epochs_stopping = 0

saved_model_name = "spacer_distr"
model_name = "BaselineCNN"
training_method = 'aux_stop_wandb'
num_outputs = 6
num_aux_outputs = 9
optimiser = 'Adamax'
lr = 0.0007                    ###0.001 #0.0007 #0.001
weight_decay = 1e-7
loss = 'MSE'
stopping_loss = 'BCE'
recon_size = 16
runConfig(transform, metrics, reach_datasets, stop_datasets)
dataset_pipe = None
































        

#****************** Data ******************#
data_folder = "distrGrasp_64"
is_shape_data = False
reach_data_mode = "aux"
stop_data_mode = "onlyStop"

n_demos = 30
train_val_split = 0.8

task = "distr"
tags_default = [f"{task}", "30_demos"]

transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


#****************** Training ******************#
epochs = 100
batch_size = 64
use_gpu = True
epochs_stopping = 40

num_outputs = 6
num_aux_outputs = 9
optimiser = 'Adamax'
lr = 0.001
weight_decay = 1e-7
loss = 'MSE'
stopping_loss = 'BCE'
recon_size = 16


################################################# Model
saved_model_name = f"BaselineCNN_{task}_{n_demos}d"
model_name = "BaselineCNN"
training_method = 'aux_stop_wandb'
tags = tags_default + ["BaselineCNN"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "currentImage"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"SpatialAE_{task}_{n_demos}d"
model_name = "SpatialAE"
training_method = 'AE_wandb'
tags = tags_default + ["SpatialAE"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "MI"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["MI_Net"]

runConfig(transform, metrics, reach_datasets, stop_datasets)


################################################# Model
saved_model_name = f"MI_Net_deeperAttention_{task}_{n_demos}d"
model_name = "MotionImage_deeper_attention"
training_method = 'AE_wandb'
tags = tags_default + ["deeper_attention"]

runConfig(transform, metrics, reach_datasets, stop_datasets)


################################################# Model
saved_model_name = f"MI_Net_auxiliary_{task}_{n_demos}d"
model_name = "MotionImage_auxiliary"
training_method = 'AE_wandb'
tags = tags_default + ["auxiliary"]

runConfig(transform, metrics, reach_datasets, stop_datasets)


################################################# Model
saved_model_name = f"MI_Net_indepAE_{task}_{n_demos}d"
model_name = "MotionImage_indepAE"
training_method = 'AE_indep'
tags = tags_default + ["indepAE"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#

reach_data_mode = "MI_64"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net64_{task}_{n_demos}d"
model_name = "MotionImage_attention_64"
training_method = 'AE_wandb'
tags = tags_default + ["MI_Net_64"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "MI_delta1"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net_delta1_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["delta_1"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "MI_unfiltered"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net_unfiltered_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["unfiltered"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "futureImage"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"Future_Net_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["future"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




################################################################################################################
################################################################################################################
################################################################################################################
dataset_pipe = None

#****************** Data ******************#
reach_data_mode = "aux"
stop_data_mode = "onlyStop"

n_demos = 70
train_val_split = 0.8

tags_default = [f"{task}", "70_demos"]

transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


#****************** Training ******************#
epochs = 100
batch_size = 64
use_gpu = True
epochs_stopping = 50

num_outputs = 6
num_aux_outputs = 9
optimiser = 'Adamax'
lr = 0.001
weight_decay = 1e-7
loss = 'MSE'
stopping_loss = 'BCE'
recon_size = 16


################################################# Model
saved_model_name = f"BaselineCNN_{task}_{n_demos}d"
model_name = "BaselineCNN"
training_method = 'aux_stop_wandb'
tags = tags_default + ["BaselineCNN"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "currentImage"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"SpatialAE_{task}_{n_demos}d"
model_name = "SpatialAE"
training_method = 'AE_wandb'
tags = tags_default + ["SpatialAE"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "MI"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["MI_Net"]

runConfig(transform, metrics, reach_datasets, stop_datasets)


################################################# Model
saved_model_name = f"MI_Net_deeperAttention_{task}_{n_demos}d"
model_name = "MotionImage_deeper_attention"
training_method = 'AE_wandb'
tags = tags_default + ["deeper_attention"]

runConfig(transform, metrics, reach_datasets, stop_datasets)


################################################# Model
saved_model_name = f"MI_Net_auxiliary_{task}_{n_demos}d"
model_name = "MotionImage_auxiliary"
training_method = 'AE_wandb'
tags = tags_default + ["auxiliary"]

runConfig(transform, metrics, reach_datasets, stop_datasets)


################################################# Model
saved_model_name = f"MI_Net_indepAE_{task}_{n_demos}d"
model_name = "MotionImage_indepAE"
training_method = 'AE_indep'
tags = tags_default + ["indepAE"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#

reach_data_mode = "MI_64"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net64_{task}_{n_demos}d"
model_name = "MotionImage_attention_64"
training_method = 'AE_wandb'
tags = tags_default + ["MI_Net_64"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "MI_delta1"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net_delta1_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["delta_1"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "MI_unfiltered"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"MI_Net_unfiltered_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["unfiltered"]

runConfig(transform, metrics, reach_datasets, stop_datasets)




#****************** Data ******************#
reach_data_mode = "futureImage"
transform, metrics, reach_datasets, stop_datasets, dataset_pipe = load_data()


################################################# Model
saved_model_name = f"Future_Net_{task}_{n_demos}d"
model_name = "MotionImage_attention"
training_method = 'AE_wandb'
tags = tags_default + ["future"]

runConfig(transform, metrics, reach_datasets, stop_datasets)