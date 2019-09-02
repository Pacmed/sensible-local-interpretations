import torch
from torch import nn
from torchvision import transforms, datasets
import numpy as np
import os
from torch.utils import data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class NN(nn.Module):
    def __init__(self, n_classes):
        super(NN, self).__init__()
        quantiles = 3
        
        n_out = 1 if n_classes == 2 else n_classes
        
        self.fe = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(8), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(8),    
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),     
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),           
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),           
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),           
            nn.ReLU(),
            nn.MaxPool2d(2),


            )
            
        self.classifier = nn.Sequential(
            nn.Linear(32, n_out),
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        z = self.fe(x)
        z = z.view(x.shape[0], -1)
        return self.classifier(z).squeeze(-1)
   

def run_test_data(model, loader):
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for data, target in tqdm(loader):
            data = data.to(model.device)
            outputs.append(model(data).cpu())
            labels.append(target.cpu())
            del data, target
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        return outputs, labels


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


def load_data(dir, batch_size):
    
        # Data loading code
    traindir = os.path.join(dir, 'train')
    valdir = os.path.join(dir, 'val')
    testdir = os.path.join(dir, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(
        datasets.ImageFolder(traindir,
                             transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    val_loader = data.DataLoader(
        datasets.ImageFolder(valdir,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True)

    # test_loader = data.DataLoader(
    #     TestImageFolder(testdir,
    #                     transforms.Compose([
    #                         transforms.Scale(256),
    #                         transforms.CenterCrop(224),
    #                         transforms.ToTensor(),
    #                         normalize,
    #                     ])),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=False)
    
    return train_loader, val_loader


def log_softmax(outputs, dim=-1, t=1):
    # Stability: https://stackoverflow.com/questions/44081007/logsoftmax-stability
    b = torch.max(outputs, dim=dim, keepdim=True)[0]
    return (outputs - b) * t - torch.log(torch.sum(torch.exp(t * (outputs - b)), dim=dim, keepdim=True))

def nll_loss(p, y, n_classes=10):
    y_oh = torch.eye(n_classes)[y].to(p.device)    
    return -1 * torch.mean(torch.sum(y_oh * torch.log(p + 1e-9), dim=-1))   


def train(model, optimizer, loss_fn, train_loader, val_loader, class_weights=None, epochs=512):
    
    best_val_loss = np.inf
    patience = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(epochs):
        
        # Train loop.
        model = model.to(device)
        model.train()
        
        train_loss = 0
        for data, target in tqdm(train_loader):
            
            data = data.to(device)
            target = target.to(device)
                        
            outputs = model(data)
            loss = loss_fn(outputs, target.float(), weight=get_weight_tensor_from_class_weights(target, class_weights))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            del data, target
            
        train_loss /= len(train_loader) 
            
        # Validation.
        model.eval()
        
        outputs, labels = run_test_data(model, val_loader)
        
        val_loss = loss_fn(outputs, labels.float(), weight=get_weight_tensor_from_class_weights(labels, class_weights)).item()
        
        auc = roc_auc_score(labels.numpy(), outputs.numpy())
        
        print(f"Epoch {epoch} \t Validation loss: {val_loss:.4f}, Train loss: {train_loss:.4f} \t AUC: {auc:.4f}")
        
        if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience = 0
                print("New best model.")
                torch.save(model.state_dict(), 'tmp.pth')
        else:
            patience += 1

            if patience == 8:
                break
                
    model.load_state_dict(torch.load('tmp.pth'))
    return model
            
            
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


def cross_entropy(y_true: np.array, y_score: np.array, n_labels: int = None,
                  reduction: str = None) -> np.array:
    """Calculate cross entropy between y_true and y_score
    Parameters
    ----------
    y_true: np.array
        Labels
    y_score: np.array
        Predictions
    n_labels: int (optional)
        Only needs to be set in multiclass case.
    reduction: str
        If and how the cross entropies should be reduced. Options: ['sum', 'mean']
    Returns
    -------
    np.array
        Cross entropy losses of the predictions
    """
    # Squeeze the array if possible.
    try:
        y_score = y_score.squeeze(axis=-1)
    except ValueError:
        pass

    # If the array has 2 dimensions treat as single-class.
    try:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
    except IndexError:
        pass

    # Calculate binary cross entropy or multiclass cross entropy.
    if len(y_score.shape) == 1:

        if y_true.shape != y_score.shape:
            raise ValueError("y_true shape not equal to y_score shape")

        ce = -(y_true * np.log2(y_score + 1e-9) + (1 - y_true) * np.log2(1 - y_score + 1e-9))

    else:
        # One-hot encode y_true
        y_true = np.eye(n_labels)[y_true]

        if y_true.shape != y_score.shape:
            raise ValueError("y_true shape not equal to y_score shape")

        ce = -np.sum(y_true * np.log2(y_score + 1e-9), axis=-1)

    # Apply reduction.
    if reduction == 'sum':
        return ce.sum()
    if reduction == 'mean':
        return ce.mean()

    return ce


def make_performance_uncertainty_plot(y_true: np.array,
                                      y_pred: np.array,
                                      y_unc: np.array,
                                      y_axis_label: str,
                                      performance_fn: callable = cross_entropy,
                                      performance_fn_args: dict = None) -> plt.figure:
    """Create plot how the uncertainty relates to model performance.
    Parameters
    ----------
    y_true: np.array
        True labels
    y_pred: np.array
        Predictions
    y_unc: np.array
        Uncertainties
    y_axis_label: str
        plot Y-axis label
    performance_fn: callable
        Performance function used
    performance_fn_args: dict
        Arguments passed to performance function
    Returns
    -------
    plt.figure
        Plot
    """
    try:
        y_unc.squeeze(-1)

    except ValueError:
        pass

    if y_unc.ndim == 2:
        y_unc = y_unc.mean(-1)

    elif y_unc.ndim > 2:
        raise ValueError(f"Invalid uncertainty shape: {y_unc.shape}")

    if y_true.ndim != 1:
        raise ValueError("Y-true not one-dimensional")

    # Placeholder
    if performance_fn_args is None:
        performance_fn_args = {}

    order = y_unc.argsort()

    sorted_uncertainties = y_unc[order]
    sorted_labels = y_true[order]
    sorted_predictions = y_pred[order]

    # Get the first index where both 0's and 1's have occurred with at least a batch size of 64.
    first_index = max(128,
                      np.argwhere(sorted_labels != sorted_labels[0])[0][0])
    performances = []
    percentages = []

    for i in range(first_index + 1, len(sorted_uncertainties)):
        selected_labels = sorted_labels[:i]
        selected_predictions = sorted_predictions[:i]

        percentages.append(100 * len(selected_predictions) / len(y_pred))

        performances.append(performance_fn(selected_labels, selected_predictions,
                                           **performance_fn_args))

    fig = plt.figure()
    sns.lineplot(percentages, performances)
    plt.xlabel('% of Uncertain Data')
    plt.ylabel(y_axis_label)
    return fig
