import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms

def load_data():
    train_dataset = CIFAR10('../../data/CIFAR10/', download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10('../../data/CIFAR10/', download=True, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    test_data, test_labels = next(iter(test_loader))
    
    return train_loader, test_data, test_labels

def log_softmax(outputs, dim=-1, t=1):
    # Stability: https://stackoverflow.com/questions/44081007/logsoftmax-stability
    b = torch.max(outputs, dim=dim, keepdim=True)[0]
    return (outputs - b) * t - torch.log(torch.sum(torch.exp(t * (outputs - b)), dim=dim, keepdim=True))


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        quantiles = 3
        self.fe = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.MaxPool2d(2),
        )
        self.quantile_preds = nn.ModuleList([nn.Linear(256, 10) for _ in range(quantiles)])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        z = self.fe(x).view(x.shape[0], -1)
        return [self.quantile_preds[i](z) for i in range(len(self.quantile_preds))]
    