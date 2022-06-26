import yaml
from Learning.train_model import model_training

def useConfig():
    configs = {}
    configs["model_name"] = model_name
    configs["data_folder"] = data_folder
    configs["n_epochs"] = epochs
    configs["batch_size"] = batch_size
    configs["training_method"] = training_method
    configs["use_gpu"] = use_gpu
    configs["optimiser"] = optimiser
    configs["learning_rate"] = lr
    configs["weight_decay"] = weight_decay
    configs["loss"] = loss
    configs["stopping_loss"] = stopping_loss
    configs["num_outputs"] = num_outputs
    configs["num_aux_outputs"] = num_aux_outputs
    configs["reconstruction_size"] = recon_size

    model_training(
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

    print("Uploading configuration details")
    saveConfig(configs)
    print("Done")

def saveConfig(configs):
    with open("Learning/TrainedModels/model_config.yaml", 'r+') as file:
        yaml.safe_load(file)
        file.write("\n")
        model_config = {f"{saved_model_name}": configs}
        yaml.dump(model_config, file, sort_keys=False)
        

#Saving and Training info
data_folder = "followDummy_3"
saved_model_name = "StrengthSpatialAE_fc_follow_2"
model_name = "StrengthSpatialAE_fc"
training_method = 'AE'

#Training process
epochs = 100
batch_size = 64
use_gpu = True

#Model parameters
num_outputs = 6
num_aux_outputs = 9

#Optimiser
optimiser = 'Adamax'
lr = 0.001
weight_decay = 1e-7

#Loss
loss = 'MSE'
stopping_loss = 'BCE'

#For Auto Encoder
recon_size = 16


useConfig()