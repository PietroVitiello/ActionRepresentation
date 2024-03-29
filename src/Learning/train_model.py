from typing import List
import torch
from torchvision import transforms
from os.path import dirname, join, abspath

from .TrainLoaders.trainloader import SimDataset
from .Training.train import Training
from .utils.utils_pipeline import model_choice, uselessParams, return_data_stats, get_trainer, dataset_pipeline #, train_model
from .utils.utils_dataloader import getTrainLoader, getTransformation

from torch.utils.data import DataLoader

def model_training(
    data_folder,
    saved_model_name,
    epochs = 100,
    stopping_epochs = 100,
    batch_size = 64,
    training_method = 'eeVel',
    dataset_modes = ["motionImage", "onlyStop"],
    use_gpu = True,
    n_demos = 100,
    train_val_split: float = 0.8,
    optimiser = 'Adamax',
    lr = 0.001,
    weight_decay = 1e-7,
    loss = 'MSE',
    stopping_loss = 'BCE',
    model_name = "BaselineCNN",
    num_outputs = 6,
    num_aux_outputs = 9,
    recon_size = 16,
    is_shape_data = False
):
    dataset_path = join(dirname(abspath(__file__)), f"../Demos/Dataset/{data_folder}/")
    
    #setup image transforms
    # mean = torch.Tensor([0.485, 0.456, 0.406])
    # std = torch.Tensor([0.229, 0.224, 0.225])
    # transform = getTransformation(mean, std)
    transform = None

    # ---------------- Dataset ---------------- #
    ds = dataset_pipeline(dataset_path, train_val_split, n_demos, is_shape_data)
    # trainSet, val_dataset_reach = SimDataset.get_with_val(dataset_path, train_val_split, transform, dataset_mode="motionImage", filter_stop=True, n_demos=n_demos)
    trainSet, val_dataset_reach = ds.get_dataset_with_val(transform, dataset_mode=dataset_modes[0], filter_stop=True)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=1)
    # print(len(trainLoader.dataset))

    # # trainSet_stop, val_dataset_stop = SimDataset.get_with_val(dataset_path, train_val_split, transform, dataset_mode="onlyStop", n_demos=n_demos)
    trainSet_stop, val_dataset_stop = ds.get_dataset_with_val(transform, dataset_mode=dataset_modes[1])
    trainLoader_stop = DataLoader(trainSet_stop, batch_size=batch_size, shuffle=True, num_workers=1)
    # trainLoader_stop = None
    # print(len(trainLoader_stop.dataset))

    val_dataloader_reach = DataLoader(val_dataset_reach, batch_size=batch_size, num_workers=1)
    val_dataloader_stop = DataLoader(val_dataset_stop, batch_size=batch_size, num_workers=1)
    val_dataloaders = (val_dataloader_reach, val_dataloader_stop)
    # print(len(val_dataloader_stop.dataset))


    transform, metrics = trainSet.get_transforms(get_stats=True)
    metrics = return_data_stats(metrics)

    # ---------------- Training ---------------- #
    torch.cuda.empty_cache()
    model = model_choice(model_name, num_outputs, num_aux_outputs, recon_size)
    training = get_trainer(training_method, model, saved_model_name, trainLoader, val_dataloaders, trainLoader_stop, transform, use_gpu, epochs, stopping_epochs, batch_size, optimiser, lr, weight_decay, loss, stopping_loss, recon_size)
    # training = Training.get_trainer(model, saved_model_name, trainLoader, val_dataloaders, trainLoader_stop, transform, use_gpu, epochs, stopping_epochs, batch_size, optimiser, lr, weight_decay, loss, stopping_loss, recon_size)
    # train_model(training, training_method)
    training.train()
    print("Training Done \n")

    # save the model
    print(f"Saving the model as {saved_model_name}")
    save_dir = join(dirname(abspath(__file__)), f'TrainedModels/{saved_model_name}.pt')
    torch.save(model.state_dict(), save_dir)
    print(f"Done\n")

    return uselessParams(training_method), metrics


def loading_data(
    data_folder,
    dataset_modes = ["motionImage", "onlyStop"],
    n_demos = 100,
    train_val_split: float = 0.8,
    dataset_pipe: dataset_pipeline = None,
    is_shape_data = False
):
    dataset_path = join(dirname(abspath(__file__)), f"../Demos/Dataset/{data_folder}/")
    transform = None

    # ---------------- Dataset ---------------- #
    if dataset_pipe is None:
        ds = dataset_pipeline(dataset_path, train_val_split, n_demos, is_shape_data)
    else:
        ds = dataset_pipe
    trainSet, val_dataset_reach = ds.get_dataset_with_val(transform, dataset_mode=dataset_modes[0], filter_stop=True)
    trainSet_stop, val_dataset_stop = ds.get_dataset_with_val(transform, dataset_mode=dataset_modes[1])

    transform, metrics = trainSet.get_transforms(get_stats=True)
    metrics = return_data_stats(metrics)
    return transform, metrics, (trainSet, val_dataset_reach), (trainSet_stop, val_dataset_stop), ds

def training_individual(
    reach_datasets,
    stop_datasets,
    transform,
    saved_model_name,
    epochs = 100,
    stopping_epochs = 100,
    batch_size = 64,
    training_method = 'eeVel',
    use_gpu = True,
    optimiser = 'Adamax',
    lr = 0.001,
    weight_decay = 1e-7,
    loss = 'MSE',
    stopping_loss = 'BCE',
    model_name = "BaselineCNN",
    num_outputs = 6,
    num_aux_outputs = 9,
    recon_size = 16,
    tags: List[str] = None
):
    # ---------------- Dataloaders ---------------- #
    trainSet, val_dataset_reach = reach_datasets
    trainSet_stop, val_dataset_stop = stop_datasets

    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader_reach = DataLoader(val_dataset_reach, batch_size=batch_size, num_workers=1)

    if trainSet_stop is not None:
        trainLoader_stop = DataLoader(trainSet_stop, batch_size=batch_size, shuffle=True, num_workers=1)
        val_dataloader_stop = DataLoader(val_dataset_stop, batch_size=batch_size, num_workers=1)
    else:
        trainLoader_stop = None
        val_dataloader_stop = None
    
    val_dataloaders = (val_dataloader_reach, val_dataloader_stop)

    # ---------------- Training ---------------- #
    torch.cuda.empty_cache()
    model = model_choice(model_name, num_outputs, num_aux_outputs, recon_size)
    training = get_trainer(training_method, model, saved_model_name, trainLoader, val_dataloaders, trainLoader_stop, transform, use_gpu, epochs, stopping_epochs, batch_size, optimiser, lr, weight_decay, loss, stopping_loss, recon_size, tags)
    # training = Training.get_trainer(model, saved_model_name, trainLoader, val_dataloaders, trainLoader_stop, transform, use_gpu, epochs, stopping_epochs, batch_size, optimiser, lr, weight_decay, loss, stopping_loss, recon_size)
    # train_model(training, training_method)
    training.train()
    print("Training Done \n")

    # save the model
    print(f"Saving the model as {saved_model_name}")
    save_dir = join(dirname(abspath(__file__)), f'TrainedModels/{saved_model_name}.pt')
    torch.save(model.state_dict(), save_dir)
    print(f"Done\n")

    return uselessParams(training_method), training.get_run_id()










#IMPORTANT GENERAL STUFF
# EPOCHS = 100
# BATCH_SIZE = 64
# LR = 0.001
# WD = 1e-7
# USE_GPU = True
# PATH_DATASET = "../../Demos/Dataset/followDummy_1/"


# #setup image transforms
# mean = torch.Tensor([0.485, 0.456, 0.406])
# std = torch.Tensor([0.229, 0.224, 0.225])

# #need to transform and need to normalize after
# transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(mean.tolist(), std.tolist())
#         ]
#     )

# # ---------------- Dataset ---------------- #
# trainSet = SimDataset(PATH_DATASET, transform)
# trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=1)

# # ---------------- Train ---------------- #
# if USE_GPU and torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print("Using GPU to train on")
# else:
#     device = torch.device('cpu')

# print_every = 10
# dtype = torch.float32
# def train_model(model,optimizer,epochs=1):
#     model = model.to(device=device)
#     mseLoss = nn.MSELoss()
#     for e in range(epochs):
#         for t, (x, ee_v) in enumerate(trainLoader):
#             model.train()
#             x = x.to(device=device,dtype=dtype)
#             ee_v = ee_v.to(device=device,dtype=dtype)

#             out = model(x)
#             loss = mseLoss(out, ee_v)

#             optimizer.zero_grad()

#             loss.backward()

#             optimizer.step()

#             if t % print_every == 0:
#                 print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))



# ---------------- Run Training ---------------- #
# torch.cuda.empty_cache()
# model = BaselineCNN(3)
# optimizer = optim.Adamax(model.parameters(), lr=LR, weight_decay=WD)

# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total number of parameters is: {}".format(params))
# train_model(model, optimizer, epochs = EPOCHS)

# # save the model
# torch.save(model.state_dict(), 'TrainedModels/baselineCNN_follow.pt')