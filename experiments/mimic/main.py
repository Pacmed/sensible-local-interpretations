import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models import MLP
from src.utils import PERFORMANCE_DICT, get_performance_metrics
import copy
from typing import Union
import pandas as pd
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from torch.nn import functional as F
import os


def get_dataloader_from_tensors(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                                shuffle: bool = True) -> DataLoader:
    """Obtain a dataloader for training/testing from input Tensors.

    Parameters
    ----------
    X: torch.Tensor
        data
    y: torch.Tensor
        labels
    batch_size: int
        batch_size
    shuffle: optional, bool
        Whether or not to shuffle the data first.

    Returns
    -------
    DataLoader
    """

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size, shuffle)

    return loader

def _convert_to_tensor(X: Union[pd.Series, pd.DataFrame, np.ndarray]) -> torch.Tensor:
     """Convert pandas instance or numpy array to torch.Tensor

     Parameters
     ----------
     X: Input dataframe or series.

     Returns
     -------
     torch.Tensor
         Converted to torch.Tensor
     """

     # Convert dataframes to numpy arrays.
     if isinstance(X, (pd.DataFrame, pd.Series)):
         X = X.values

     # Convert arrays to Tensors.
     X = torch.from_numpy(X.astype('float32'))
     return X

def test(model, epoch, train_loader, test_loader, verbose=True,
             save_models: bool = False):
    """Test the BNN

    Parameters
    ----------
    epoch: int
        current epoch
    train_loader: torch.utils.DataLoader
        not used here, but kept for consistency with other BNNs and BNNClassifier
    test_loader: torch.utils.data.DataLoader
    writer: SummaryWriter
    verbose: bool
        Print statements
    save_models: bool
        Save model at end of epoch.
    kwargs: dict
        For compatibility with base class.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    performance_dict = copy.deepcopy(PERFORMANCE_DICT)
    num_batches = len(test_loader)

    with torch.no_grad():

        for batch_idx, (x, y) in enumerate(tqdm(test_loader, position=1)):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = model.nll(logits, y)

            if model.n_classes == 1:

                probits = torch.sigmoid(logits)

            else:
                probits = F.softmax(logits, dim=-1)

            auc, f1, ap, acc = get_performance_metrics(y.numpy(), probits.numpy())

            performance_dict['loss'] += loss.item()
            performance_dict['nll'] += loss.item()
            performance_dict['accuracy'] += acc
            performance_dict['auc'] += auc
            performance_dict['f1_score'] += f1
            performance_dict['average_precision'] += ap

    for k in performance_dict:
        performance_dict[k] /= num_batches

    if save_models:
        if not os.path.exists('./models'):
            os.makedirs('./models', exist_ok=True)

        save_path = os.path.join('./models', f'full_{epoch}.pth')

        torch.save(model.state_dict(), save_path)

    if verbose:
        performance_str = f"Epoch: {epoch} | " \
            f"Total Loss: {round(performance_dict['loss'], 2)} | " \

        print(performance_str)

    return performance_dict

def train(model, optimizer, train_loader):
    """Train the BNN

    Parameters
    ----------
    optimizer: torch.Optimizer
    train_loader: torch.utils.data.DataLoader
    epoch: int
        Current epoch
    writer: SummaryWriter
        SummaryWriter instance.
    kwargs: dict
        For compatibility with base class.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for batch_idx, (x, y) in enumerate(tqdm(train_loader, position=1)):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        outputs = model(x)

        outputs = outputs.squeeze(-1)
        loss = model.nll(outputs, y.squeeze(-1))
        loss.backward()
        optimizer.step()


def fit(model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        lr: float = 1e-3,
        alpha: float = 1,
        validate: bool = False,
        X_val: Union[np.ndarray, pd.DataFrame] = None,
        y_val: Union[np.ndarray, pd.Series] = None,
        batch_size=128,
        val_batch_size=128,
        early_stopping_patience=8,
        epochs=256,
        verbose: bool = True):
    """Fit the BNN to a dataset.

    Parameters
    ----------
    X: Union[np.ndarray, pd.DataFrame]
        Training data.
    y: Union[np.ndarray, pd.Series]
        Training labels.
    lr: float
        Learning rate.
    alpha: float
        The size of the prediction loss compared to the KL-loss.
    validate: bool
        Whether or not to run validation to monitor performance.
    X_val: Union[np.ndarray, pd.DataFrame],  optional
        Validation data
    y_val: Union[np.ndarray, pd.DataFrame], optional
        Validation labels
    verbose: bool
        Print performance metrics?

    Returns
    -------
    BNNClassifier
    """

    # Convert inputs to tensors.
    X, y = _convert_to_tensor(X), _convert_to_tensor(y)

    X = X.reshape(X.shape[0], -1)

    if validate:
        if X_val is None or y_val is None:
            raise ValueError("No validation data provided!")

        X_val, y_val = _convert_to_tensor(X_val), _convert_to_tensor(y_val)
        X_val = X_val.reshape(X_val.shape[0], -1)

    if model.n_classes > 1:
        y, y_val = y.long(), y_val.long()

    # Get the optimizer.
    # optimizer = Adam(model.parameters(), lr=lr)
    optimizer = SGD(model.parameters(), lr=1e-3)

    # Initialize dataloader.
    train_loader = get_dataloader_from_tensors(X, y, batch_size)

    if validate:
        test_loader = get_dataloader_from_tensors(X_val, y_val, val_batch_size)

    if early_stopping_patience is not None:
        best_loss = 1e9  # Large number placeholder
        patience = 0

    # Train
    for epoch in tqdm(range(epochs), position=0):
        train(model, optimizer, train_loader)

        if validate:
            performance_dict = test(model, epoch, train_loader, test_loader,
                                                 verbose=verbose)

            if early_stopping_patience is not None:

                if round(performance_dict['loss'], 2) <= round(best_loss, 2):
                    best_loss = performance_dict['loss']
                    patience = 0
                    best_parameters = model.state_dict()
                    torch.save(best_parameters, 'tmp.pth')

                else:
                    patience += 1

                if patience == early_stopping_patience:
                    print("Loading saved parameters..")
                    model.eval()
                    model.load_state_dict(torch.load('tmp.pth'))
                    os.remove('tmp.pth')
                    break

    return model

