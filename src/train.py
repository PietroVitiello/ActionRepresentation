import yaml
import numpy as np
from Learning.train_model import model_training

def runConfig():
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

    print(f"\nTraining {model_name} Model\n")
    print(f"{int(np.round(n_demos * (2-train_val_split)))} Total Demonstrations: {n_demos} T, {int(np.round(n_demos * (1-train_val_split)))} V")
    useless_keys, metrics = model_training(
                                data_folder,
                                saved_model_name,
                                epochs,
                                epochs_stopping,
                                batch_size,
                                training_method,
                                use_gpu,
                                n_demos,
                                train_val_split,
                                optimiser,
                                lr,
                                weight_decay,
                                loss,
                                stopping_loss,
                                model_name,
                                num_outputs,
                                num_aux_outputs,
                                recon_size
                            )
    keepUseful(configs, useless_keys)
    configs["input_metrics"] = [metrics[0][0].tolist(), metrics[0][1].tolist()]
    if metrics[1][0] is not None:
        configs["recon_metrics"] = [metrics[1][0].tolist(), metrics[1][1].tolist()]
    configs["eeVel_metrics"] = [metrics[2][0].tolist(), metrics[2][1].tolist()]
    configs["aux_metrics"] = [metrics[3][0].tolist(), metrics[3][1].tolist()]      

    print("Uploading configuration details")
    saveConfig(configs)
    print("Configurations saved in 'Learning/TrainedModels/model_config.yaml'\n")
    

def saveConfig(configs):
    with open("Learning/TrainedModels/model_config.yaml", 'r+') as file:
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
        

#Saving and Training info
data_folder = "linearGrasp_experiment_64"
saved_model_name = "ReduceTo1x1_100_100_reach_noNorm"
model_name = "ReduceTo1x1"
training_method = 'eeVel_aux_wandb'

#Training process
epochs = 100
batch_size = 64
use_gpu = True
n_demos = 100
train_val_split = 0.8

#Model parameters
num_outputs = 6
num_aux_outputs = 9

#Optimiser
optimiser = 'Adamax'
lr = 0.0007                  ###0.001 #0.0007 #0.001
weight_decay = 1e-7

#Loss
loss = 'MSE'
stopping_loss = 'BCE'

#For Auto Encoder
recon_size = 16

#For individual stopping
epochs_stopping = 30

runConfig()