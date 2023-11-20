from sklearn.datasets import make_moons
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def get_dataset(n_samples: int, batch_size: int):
    X, y = make_moons(n_samples=n_samples)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset=dataset, batch_size=batch_size)

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