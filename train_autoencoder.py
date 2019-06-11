import time
import os
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as td
import pandas as pd
from models import SegNet
from utils import *
from PIL import Image
import torchvision as tv
def train(epoch, train_loader, model, optimizer):
    # Ensure dropout layers are in train mode
    model.train()

    batch_time = ExpoAverageMeter()  
    losses = ExpoAverageMeter() 

    start = time.time()
    train_loss = []
    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # Set device options
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        y_hat = model(x)
        loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        train_loss.append(losses.val)
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))
    return train_loss


def valid(val_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = ExpoAverageMeter()  
    losses = ExpoAverageMeter()  
    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            
            loss = torch.sqrt((y_hat - y).pow(2).mean())
            
            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    return losses.avg


class SDODataset(td.Dataset):
    def __init__(self, root_dir, mode="train", image_size=(224, 224)):
        super(SDODataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.df = pd.read_csv(root_dir + mode + '/meta_data.csv', sep=",", parse_dates=["start", "end"], index_col="id")
        self.image_path = []
        self.label = []
        self.id = []
        for row in self.df.iterrows():
            ar_nr, p = row[0].split("_", 1)
            img_path = os.path.join(root_dir, mode, ar_nr, p)
            for img_name in os.listdir(img_path):
                if img_name.endswith('_magnetogram.jpg'):
                    self.image_path.append(os.path.join(img_path, img_name))
                    self.label.append(row[1]['peak_flux'])
                    self.id.append(row[0])
    def __len__(self):
        return len(self.label)
    def __repr__(self):
        return "SDODataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        img = Image.open(img_path)
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            ])
        x = transform(img)
        d = self.label[idx]
        return x, x
    def number_of_classes(self):
        return self.data['class'].max() + 1

def main():
    Dataset = './SDOBenchmark-data-full/'
    train_set = SDODataset(Dataset, mode = 'train')
    test_set = SDODataset(Dataset, mode = 'test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    # Create SegNet model
    label_nbr = 1
    model = SegNet(label_nbr,in_channels=1)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # Use appropriate device
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = 100000
    epochs_since_improvement = 0
    state, start_epoch = load_checkpoint(mode = 'autoencoder')
    if start_epoch != 0:
        print("Load from checkpoint epoch: ", start_epoch - 1)
        model = state['model']
        model = model.to(device)
        optimizer = state['optimizer']
    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss = train(epoch, train_loader, model, optimizer)

        # One epoch's validation
        val_loss = valid(val_loader, model)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best, mode = 'autoencoder',train_loss = train_loss)
    print('train finished')

if __name__ == '__main__':
    main()
