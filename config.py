import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_nbr = 1
imsize = 224
batch_size = 16
lr = 0.0001
patience = 50
start_epoch = 0
epochs = 5
print_freq = 20
regression_checkpoint = 'regression'
autoencoder_checkpoint = 'autoencoder'
