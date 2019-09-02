import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]


def load_data(n_classes=10):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(15, translate=(0.1, 0.1), 
                                scale=(0.9, 1.1), shear=None, resample=False, fillcolor=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    classes_to_keep = np.arange(n_classes)
    
    train_dataset = CIFAR10('../../data/CIFAR10/', download=True, transform=transforms)
    test_dataset = CIFAR10('../../data/CIFAR10/', 
                           download=True, train=False, transform=transforms.ToTensor())
    
    train_data = train_dataset.data
    train_labels = np.array(train_dataset.targets)
    
    test_data = test_dataset.data
    test_labels = np.array(test_dataset.targets)
    
    train_indices_to_keep = np.isin(train_labels, classes_to_keep)
    test_indices_to_keep = np.isin(test_labels, classes_to_keep)
    
    train_data = train_data[train_indices_to_keep]
    test_data = test_data[test_indices_to_keep]
    
    train_labels = train_labels[train_indices_to_keep]
    test_labels = test_labels[test_indices_to_keep]
    
    train_dataset = CustomTensorDataset((train_data, train_labels), transform=transform)
    test_dataset = CustomTensorDataset((test_data, 
                                       test_labels), 
                                       transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    test_data, test_labels = next(iter(test_loader))
    
    return train_loader, test_data, test_labels

def log_softmax(outputs, dim=-1, t=1):
    # Stability: https://stackoverflow.com/questions/44081007/logsoftmax-stability
    b = torch.max(outputs, dim=dim, keepdim=True)[0]
    return (outputs - b) * t - torch.log(torch.sum(torch.exp(t * (outputs - b)), dim=dim, keepdim=True))

def nll_loss(p, y, n_classes=10):
    y_oh = torch.eye(n_classes)[y].to(p.device)    
    return -1 * torch.mean(torch.sum(y_oh * torch.log(p + 1e-9), dim=-1))

class NN(nn.Module):
    def __init__(self, n_classes):
        super(NN, self).__init__()
        quantiles = 3
        
        n_out = 1 if n_classes == 2 else n_classes
        
        self.fe = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),           
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),           
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),           
            nn.MaxPool2d(2),
            nn.Dropout(0.2),           
            
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
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(512, n_out)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        z = self.fe(x)
        z = z.view(x.shape[0], -1)
        return self.classifier(z).squeeze(-1)
   

def train(model, optimizer, loss_fn, train_loader, test_data, test_labels, class_weights=None):
    
    best_val_loss = np.inf
    patience = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(512):
        
        # Train loop.
        model = model.to(device)
        model.train()
        
        train_loss = 0
        for data, target in train_loader:
            
            data = data.to(device)
            target = target.to(device)
                        
            outputs = model(data)
            loss = loss_fn(outputs, target.float(), weight=get_weight_tensor_from_class_weights(target, class_weights))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader) 
            
        # Validation.
        model.eval()
        model.to('cpu')
        outputs_val = model(test_data)
        val_loss = loss_fn(outputs_val, test_labels.float(), weight=get_weight_tensor_from_class_weights(test_labels, class_weights))
        
        print(f"Epoch {epoch} \t Validation loss: {val_loss.item():.2f}, Train loss: {train_loss:.2f}")
        
        if val_loss.item() < best_val_loss - 1e-4:
                best_val_loss = val_loss.item()
                patience = 0
                torch.save(model.state_dict(), 'tmp.pth')
        else:
            patience += 1

            if patience == 8:
                model.load_state_dict(torch.load('tmp.pth'))
                break
            
            
def accuracy_multiclass(model, val_data, val_target):
    model_outputs = model(val_data)
    predictions = torch.argmax(model_outputs, dim=-1)
    accuracy = (predictions == val_target).float().mean()
    print(f"Accuracy: {100 * accuracy:.2f}")
    

def get_weight_tensor_from_class_weights(y, class_weights: list):
    
    if class_weights is None:
        return None
    
    weights = torch.empty_like(y).to(y.device).float()
    
    for i, weight in enumerate(class_weights):
        
        if isinstance(weight, np.ndarray):
            weight = weight.item()
        weights[y == i] = weight
    
    return weights
