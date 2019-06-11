import os

from config import *
import itertools

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val

def load_checkpoint(mode):
    if mode == 'regression':
        folder = regression_checkpoint
    elif mode == 'autoencoder':
        folder = autoencoder_checkpoint
    else:
        print('Error load mode: please select regression or autoencoder mode.')
    ensure_folder(folder)
    max_saved_epoch = -1
    last_epoch_filename = ""
#     statr_epoch = 0
    for file_name in os.listdir(folder):
        split=file_name.split('_')
        if split[1].isdigit():
            if int(split[1]) > max_saved_epoch:
                max_saved_epoch = int(split[1])
#                 start_epoch = max_saved_epoch + 1
                last_epoch_filename = file_name
    state = {}
    if len(last_epoch_filename) != 0:
        state = torch.load(os.path.join(folder,last_epoch_filename))
    return state, max_saved_epoch + 1

def save_checkpoint(epoch, model, optimizer, val_loss, is_best,mode, train_loss):
    if mode == 'regression':
        folder = regression_checkpoint
    elif mode == 'autoencoder':
        folder = autoencoder_checkpoint
    else:
        print('Error load mode: please select regression or autoencoder mode.')
    ensure_folder(folder)
    state = {'model': model,
             'optimizer': optimizer,
             'train_loss': train_loss}
    filename = '{0}/checkpoint_{1}_{2:.3f}.tar'.format(folder, epoch, val_loss)
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, '{}/BEST_checkpoint.tar'.format(folder))
