from sklearn.datasets import make_blobs
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_dataset(n_samples: int, batch_size: int):
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=[[1, -1], [1, 1], [-1, -1], [-1, 1]], cluster_std=0.1)
    y[y == 2] = 1
    y[y == 3] = 0
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset=dataset, batch_size=batch_size), X

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def train(data_loader, n_samples, model, loss_function, optimizer, device):
    epoch_loss, epoch_acc = 0., 0
    
    for X, y in data_loader:
        
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_function(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(X)
        epoch_acc += ((pred > 0.5).type(torch.float) == y).sum().item()
        
    return epoch_loss / n_samples, epoch_acc / n_samples

def vis_losses_accs(losses, accs):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
    axes[0].plot(range(len(losses)), losses)
    axes[0].set_ylabel("BCE Loss")
    axes[1].plot(range(len(accs)), accs)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    plt.show()
    
def vis_classifications(model, X, preds, device):
    _, ax = plt.subplots(figsize=(7, 7))
    colors = lambda x: ['red' if value == 0 else 'blue' for value in x]
    ax.scatter(X[:, 0], X[:, 1], c=colors((preds > 0.5).type(torch.float)), s=10)
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    grid_X, grid_Y = np.meshgrid(x, y)
    grid_T = np.hstack((grid_X.reshape(-1, 1), grid_Y.reshape(-1, 1)))
    grid_preds = model(torch.FloatTensor(grid_T).to(device))
    ax.scatter(grid_X, grid_Y, c=colors((grid_preds > 0.5).type(torch.float)), s=10, alpha=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()