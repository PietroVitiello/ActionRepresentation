import yaml
from Learning.train_model import model_training

def runConfig():
    configs = {}
    configs["model_name"] = model_name
    configs["data_folder"] = data_folder
    configs["training_method"] = training_method
    configs["n_epochs"] = epochs
    configs["batch_size"] = batch_size
    configs["use_gpu"] = use_gpu
    configs["optimiser"] = optimiser
    configs["learning_rate"] = lr
    configs["weight_decay"] = weight_decay
    configs["loss"] = loss
    configs["stopping_loss"] = stopping_loss
    configs["num_outputs"] = num_outputs
    configs["num_aux_outputs"] = num_aux_outputs
    configs["reconstruction_size"] = recon_size

    print(f"Training {model_name} Model\n")
    useless_keys = model_training(
                    data_folder,
                    saved_model_name,
                    epochs,
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
                    recon_size
                )
    keepUseful(configs, useless_keys)

    print("Uploading configuration details")
    saveConfig(configs)
    print("Configurations saved in 'Learning/TrainedModels/model_config.yaml'\n")
    

def saveConfig(configs):
    with open("Learning/TrainedModels/model_config.yaml", 'r+') as file:
        yaml.safe_load(file)
        file.write("\n")
        configs["Testing"] = {}
        configs["Testing"]["Cube_Reached"] = []
        configs["Testing"]["Attempts"] = None
        model_config = {f"{saved_model_name}": configs}
        yaml.dump(model_config, file, sort_keys=False)
        file.write("#NOTE:\n")

def keepUseful(configs:dict, useless: list):
    for key in useless:
        configs.pop(key)
        

#Saving and Training info
data_folder = "HumanLikeDemos_1"
saved_model_name = "Aux_BaselineCNN_3"
model_name = "Aux_BaselineCNN"
training_method = 'eeVel_aux'

#Training process
epochs = 150
batch_size = 64
use_gpu = True

#Model parameters
num_outputs = 6
num_aux_outputs = 9

#Optimiser
optimiser = 'Adamax'
lr = 0.0007 #0.001
weight_decay = 1e-7

#Loss
loss = 'MSE'
stopping_loss = 'BCE'

#For Auto Encoder
recon_size = 16


runConfig()