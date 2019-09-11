from notebook_utils import load_data, train, accuracy_multiclass, run_test_data, NN
from torchsummary import summary

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from torch import nn

n_classes = 2

train_loader, val_loader = load_data('/home/davidr/projects/class-weight-uncertainty/data/cats_dogs/', batch_size=64)

test_labels = torch.cat([batch[1] for batch in val_loader], dim=0)

canonical = NN(n_classes=n_classes)
optimizer = torch.optim.Adam(canonical.parameters(), lr=1e-3)
loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
print(summary(canonical, input_size=(3, 224, 224)))
train(canonical, optimizer, loss_fn, 
      train_loader, val_loader, class_weights=[0.5, 0.5])

canonical.eval()
torch.save(canonical.state_dict(), 'canonical.pth')

overconfident = NN(n_classes=n_classes)
overconfident.eval()

for name, parameter in overconfident.fe.named_parameters():
    weight = canonical.fe.state_dict()[name]
    parameter.data = weight
    parameter.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
optimizer = torch.optim.Adam(overconfident.parameters(), lr=1e-3)
loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
overconfident_weight = [0.4, 0.6]
train(overconfident, optimizer, loss_fn, train_loader, val_loader, class_weights=overconfident_weight)

overconfident.eval()
torch.save(overconfident.state_dict(), 'overconfident.pth')

del overconfident

underconfident = NN(n_classes=n_classes)
underconfident.eval()

for name, parameter in underconfident.fe.named_parameters():
    weight = canonical.fe.state_dict()[name]
    parameter.data = weight
    parameter.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
optimizer = torch.optim.Adam(underconfident.parameters(), lr=1e-3)
loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
underconfident_weight = [0.6, 0.4]
train(underconfident, optimizer, loss_fn, train_loader, val_loader, class_weights=underconfident_weight)
underconfident.eval()
torch.save(underconfident.state_dict(), 'underconfident.pth')


