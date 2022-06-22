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
    configs["reconstruction_size"] = recon_size

    model_training(
        data_folder,
        saved_model_name,
        model_name,
        epochs,
        batch_size,
        training_method,
        use_gpu,
        optimiser,
        lr,
        weight_decay,
        loss,
        recon_size
    )
    saveConfig(configs)

def saveConfig(configs):
    with open("Learning/TrainedModels/model_config.yaml", 'r+') as file:
        dataset = {f"{saved_model_name}": configs}
        yaml.dump(dataset, file)
        file.write("\n")

#Saving info
data_folder = "followDummy_fixed_2"
saved_model_name = "SpatialAE_fc_follow_1"

#Model and Training process
model_name = "SpatialAE_fc"
epochs = 100
batch_size = 64
training_method = 'AE'
use_gpu = True

#Optimiser
optimiser = 'Adamax'
lr = 0.001
weight_decay = 1e-7

#Loss
loss = 'MSE'

#For Auto Encoder
recon_size = 16


useConfig()