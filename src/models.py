from torch import nn
import torch
from tqdm.auto import tqdm
from src.utils import write_performance_scalars, get_performance_metrics, PERFORMANCE_DICT
import copy
import os
from torch.nn import functional as F

MODEL_DIR = './models/'


class MLP(nn.Module):
    """Simple MLP with dropout option.

    Parameters
    ----------
    in_dim: int
        Number of features in
    n_classes: int
        Number of classes
    p: float
        Dropout probability.
    """

    def __init__(self,
                 in_dim: int,
                 n_classes: int,
                 hidden_dims: tuple = (32, 32),
                 p: float = 0.1,
                 class_weights: list = None):
        super(MLP, self).__init__()
        self.dropout_p: float = p

        self.architecture: list = [in_dim] + [dim for dim in hidden_dims] + [n_classes]

        self.layers: list = []

        # Add linear combinations to the forward function. So iterating up to the n-1th layer and
        # adding combinations up to the nth layer.
        for i in range(len(self.architecture) - 1):
            self.layers.append(nn.Linear(self.architecture[i], self.architecture[i + 1]))

            # Don't add these in the final linear combination.
            if i < len(self.architecture) - 2:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=self.dropout_p))

        self.f: nn.Sequential = nn.Sequential(*self.layers)
        self.n_classes: int = n_classes
        self.in_dim: int = in_dim
        self.n_samples = 512

        weight_tensor = None  # Placeholder
        if self.n_classes == 1:
            if class_weights is not None:
                weight_tensor = torch.Tensor([class_weights[1]])
            self.nll = nn.BCEWithLogitsLoss(pos_weight=weight_tensor, reduction='sum')

        else:
            if class_weights is not None:
                weight_tensor = torch.Tensor(class_weights)
            self.nll = nn.CrossEntropyLoss(weight=weight_tensor, reduction='sum')

    def set_n_samples(self, n_samples):
        """Set the number of samples from the posterior in the forward pass.

        Parameters
        ----------
        n_samples: int

        Returns
        -------
        torch.Module
        """
        self.n_samples = n_samples
        return self

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Pytorch forward funcion.

        Parameters
        ----------
        input_tensor: torch.Tensor
            input tensor.

        Returns
        -------
        output_tensor: torch.Tensor
            output tensor.
        """
        if len(input_tensor.shape) > 2:
            x: torch.Tensor = input_tensor.view(-1, self.in_dim)
        else:
            x = input_tensor
        output_tensor: torch.Tensor = self.f(x)
        return output_tensor

    def fit(self, optimizer, epoch, train_loader, writer=None):
        """Train the BNN

        Parameters
        ----------
        optimizer: torch.Optimizer
        train_loader: torch.utils.data.DataLoader
        epoch: int
            Current epoch
        writer: SummaryWriter
            SummaryWriter instance.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.train()

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, position=1)):
            x, y = x.to(device), y.to(device)
            self.zero_grad()

            outputs = self(x)

            outputs = outputs.squeeze(-1)
            loss = self.nll(outputs, y)

            loss.backward()
            optimizer.step()

            performance_dict = {
                'loss': loss.item(),
                'nll': loss.item()
            }

            step = epoch * len(train_loader) + batch_idx

            if writer:
                write_performance_scalars(writer, './logs/', step, performance_dict)

        return True

    def validate(self, epoch, train_loader, test_loader, writer=None, verbose=False,
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
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.train()

        performance_dict = copy.deepcopy(PERFORMANCE_DICT)
        num_batches = len(test_loader)

        with torch.no_grad():

            for batch_idx, (x, y) in enumerate(tqdm(test_loader, position=1)):
                x, y = x.to(device), y.to(device)

                outputs = self(x)

                logits = outputs.squeeze(-1)

                loss = self.nll(logits, y)

                if self.n_classes == 1:
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

        if writer:
            step = epoch
            write_performance_scalars(writer, 'val_logs', step, performance_dict)

        if save_models:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR, exist_ok=True)

            save_path = os.path.join(MODEL_DIR, f'full_{epoch}.pth')

            torch.save(self.state_dict(), save_path)

        if verbose:
            performance_str = f"Epoch: {epoch} | " \
                f"Total Loss: {round(performance_dict['loss'], 4)} | "
            print(performance_str)

        return performance_dict
