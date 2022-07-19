import torch
from torchvision import transforms
from os.path import dirname, join, abspath

from .TrainLoaders.trainloader import SimDataset
from .training import Train
from .utils.utils_pipeline import model_choice, train_model, uselessParams
from .utils.utils_dataloader import getTrainLoader

from torch.utils.data import DataLoader

def model_training(
    data_folder,
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
    recon_size = 16
):
    dataset_path = join(dirname(abspath(__file__)), f"../Demos/Dataset/{data_folder}/")
    
    #setup image transforms
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    #need to transform and need to normalize after
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ]
        )

    # ---------------- Dataset ---------------- #
    trainSet = SimDataset.get(dataset_path, transform, dataset_mode="aux", filter_stop=True)
    # trainSet = SimDataset(dataset_path, transform, dataset_mode="aux")
    trainLoader = getTrainLoader(trainSet, batch_size=batch_size, model=model_name)

    trainSet_stop = SimDataset(dataset_path, transform, dataset_mode="onlyStop")
    trainLoader_stop = DataLoader(trainSet_stop, batch_size=batch_size, shuffle=True, num_workers=1)

    # ---------------- Training ---------------- #
    torch.cuda.empty_cache()
    model = model_choice(model_name, num_outputs, num_aux_outputs, recon_size)
    training = Train(model, trainLoader, trainLoader_stop, use_gpu, epochs, stopping_epochs, batch_size, optimiser, lr, weight_decay, loss, stopping_loss, recon_size)
    train_model(training, training_method)
    print("Training Done \n")

    # save the model
    print(f"Saving the model as {saved_model_name}")
    save_dir = join(dirname(abspath(__file__)), f'TrainedModels/{saved_model_name}.pt')
    torch.save(model.state_dict(), save_dir)
    print(f"Done\n")

    return uselessParams(training_method)










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