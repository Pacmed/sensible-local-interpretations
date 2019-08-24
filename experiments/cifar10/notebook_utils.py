import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms

def load_data():
<<<<<<< HEAD
    
    transform = transforms.Compose([
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None, resample=False, fillcolor=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    train_dataset = CIFAR10('../../data/CIFAR10/', download=True, transform=transform)
    test_dataset = CIFAR10('../../data/CIFAR10/', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    
    return train_loader, test_loader
=======
    train_dataset = CIFAR10('../../data/CIFAR10/', download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10('../../data/CIFAR10/', download=True, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    test_data, test_labels = next(iter(test_loader))
    
    return train_loader, test_data, test_labels
>>>>>>> ed90ca234b9073b2ad26627bf79422565d39b7fb

def log_softmax(outputs, dim=-1, t=1):
    # Stability: https://stackoverflow.com/questions/44081007/logsoftmax-stability
    b = torch.max(outputs, dim=dim, keepdim=True)[0]
    return (outputs - b) * t - torch.log(torch.sum(torch.exp(t * (outputs - b)), dim=dim, keepdim=True))


<<<<<<< HEAD
def nll_loss(p, y, n_classes=10):
    y_oh = torch.eye(n_classes)[y].to(p.device)    
    
    return -1 * torch.mean(torch.sum(y_oh * torch.log(p + 1e-9), dim=-1))


=======
>>>>>>> ed90ca234b9073b2ad26627bf79422565d39b7fb
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        quantiles = 3
        self.fe = nn.Sequential(
<<<<<<< HEAD
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),           
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),           
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),           
            nn.MaxPool2d(2),
            nn.Dropout(0.2)           
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(1024, 10)
        )
=======
        nn.Conv2d(3, 16, kernel_size=3),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.MaxPool2d(2),
        )
        self.quantile_preds = nn.ModuleList([nn.Linear(256, 10) for _ in range(quantiles)])
>>>>>>> ed90ca234b9073b2ad26627bf79422565d39b7fb
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
<<<<<<< HEAD
        z = self.fe(x)
        z = z.view(x.shape[0], -1)
        return self.classifier(z)
=======
        z = self.fe(x).view(x.shape[0], -1)
        return [self.quantile_preds[i](z) for i in range(len(self.quantile_preds))]
    
>>>>>>> ed90ca234b9073b2ad26627bf79422565d39b7fb
