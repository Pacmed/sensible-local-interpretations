from torch import nn
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
# from pacmagic_deeplearning.modeling.classifiers.bayesian.train_utils import make_performance_uncertainty_plot, cross_entropy
import seaborn as sns


def get_weight_tensor_from_class_weights(y, class_weights: list):
    weights = torch.empty_like(y).float()
    
    for i, weight in enumerate(class_weights):
        
        if isinstance(weight, np.ndarray):
            weight = weight.item()
        weights[y == i] = weight
    
    return weights


class NNDropout(nn.Module):
    def __init__(self, quantiles):
        super(NNDropout, self).__init__()
        self.z = nn.Linear(44, 64)
        self.bn = nn.BatchNorm1d(64)
        self.quantile_preds = nn.ModuleList([nn.Linear(64, 1) for _ in range(len(quantiles))])
        
    def forward(self, x):
        z = self.bn(self.z(x))
        
        return [self.quantile_preds[i](z) for i in range(len(self.quantile_preds))]
    
    
class NN(nn.Module):
    def __init__(self, quantiles):
        super(NN, self).__init__()
        self.z = nn.Linear(44, 64)
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(64)
        self.quantile_preds = nn.ModuleList([nn.Linear(64, 1) for _ in range(len(quantiles))])
        
    def forward(self, x):
        z = self.dropout(self.bn(self.z(x)))
        
        return [self.quantile_preds[i](z) for i in range(len(self.quantile_preds))]
    

def train(model, data, target, data_val, target_val, optimizer, quantiles, epsilon=1e-3):
    
    best_val_loss = np.inf
    patience = 0
    
    for epoch_idx in range(10000):
        model.train()
        outputs = model(data)
        
        losses = [torch.nn.functional.binary_cross_entropy_with_logits(
            outputs[i], 
            target, 
            weight=get_weight_tensor_from_class_weights(target, [quantiles[i], 1 - quantiles[i]])
        ) for i in range(len(outputs))]
        
        loss = sum(losses)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch_idx % 10 == 0:
            model.eval()
            outputs_val = model(data_val)
            losses_val = [torch.nn.functional.binary_cross_entropy_with_logits(
            outputs_val[i], 
            target_val, 
            weight=get_weight_tensor_from_class_weights(target_val, [quantiles[i], 1 - quantiles[i]])
            ) for i in range(len(outputs))]
        
            loss_val = sum(losses_val) 
            
            print(f"Validation loss: {loss_val.item():.2f}, Train loss: {loss.item():.2f}")
            
            # print(model.z.weight[0])
                        
            if loss_val.item() < best_val_loss - 1e-4:
                best_val_loss = loss_val.item()
                patience = 0
                torch.save(model.state_dict(), 'tmp.pth')
            else:
                patience += 1
                
                if patience == 8:
                    model.load_state_dict(torch.load('tmp.pth'))
                    return model 
                

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

                
                
def plot_calibration_curve(y_true, y_prob,
                           y_min: float = 0,
                           y_max: float = 1,
                           n_bins: int = 5,
                           n_std: int = 4,
                           y_std=None,
                           n: int = None,
                           z: float = 1.645,
                           ):
    """Plot the calibration curve. This curve shows if predicted probabilities and observed
    frequencies are inline. For example, if well calibrated 100 observations with y_pred = 0.1
    should contain 10 observation with y_true = 1

    Parameters
    ----------
    y_true : array-like
        Target value of y.
    y_prob : array-like
        Predicted probability values of y.
    y_min : float, optional
        Minimum value of the value axis.
    y_max : float, optional
        Maximum value of the value axis, y_pred > y_max will be maximized to y_max.
    n_bins : int, optional
        Number of bins of the histograms.
    n_std : int, optional
        Number of standard deviations to include.
    y_std: array-like, optional
        If you have a Bayesian standard deviation around your predictions
    n: int, optional
        Number of samples taken (in Bayesian Case)
    z: float, optional
        Z value for confidence interval.

    Returns
    -------
    type: matplotlib.figure.Figure
        A plot

    """

    # Make dataframe with y_true and y_pred
    df_plot = pd.DataFrame(y_prob)
    df_plot.columns = ['y_prob']
    df_plot['y_true'] = y_true
    # Initialize plot
#     # Make histogram of predicted probabilities
#     x = df_plot['y_prob'].values
#     x = np.maximum(np.minimum(x, y_max), y_min)
#     ax1.hist(x, bins=n_bins, color=SKYBLUE_COLOR)

    # Overlay the calibration curve
    mean = df_plot['y_prob'].mean()
    std = df_plot['y_prob'].std()

    # Make bins for the calibration
    df_plot['group'] = df_plot['y_prob'].apply(lambda a: np.round((a - mean) / std))
    df_plot['group'] = np.minimum(np.maximum(df_plot['group'], -n_std), n_std)
    df_plot['group'] = df_plot['group'].apply(lambda a: mean + std * a)
    df_plot['group'] = np.maximum(np.minimum(df_plot['group'], y_max), y_min)

    
    df_agg = df_plot.groupby('group')['y_true', 'y_prob'].mean().reset_index()

    y_u = None
    y_l = None

    x = df_agg['y_prob']
    y = df_agg['y_true']

    # Plot the calibration curve
    return x, y

def loss_at_50(y_true, y_prob, y_unc):
    
    # loss at 50%
    y_50 = y_true[y_unc.argsort()[:int(0.5 * len(y_unc))]]
    y_prob_50 = y_prob[y_unc.argsort()[:int(0.5 * len(y_unc))]]

    return cross_entropy(y_50, y_prob_50, reduction='sum')
    
    
    
def make_precision_accuracy_plot(outputs_val, preds, y_test):

    uppers = []
    lowers = []
    accuracies = []

    for i in range(0, 1 + int(outputs_val.shape[1] / 2)):
        j = outputs_val.shape[1] - i

        uppers.append(outputs_val[:, i].mean())
        lowers.append(outputs_val[:, j - 1].mean())

        accuracy_in_range = (preds[:, i: j] == np.repeat(y_test.values, j - i, axis=1)).any(axis=1).mean()   


        accuracies.append(accuracy_in_range)


    fig, ax1 = plt.subplots()


    ax1.fill_between(range(0, 1 + int(outputs_val.shape[1] / 2)), lowers, uppers, alpha=0.1)
    ax1.set_ylabel('Precision')

    ax2 = ax1.twinx()
    ax2.plot(range(0, 1 + int(outputs_val.shape[1] / 2)), accuracies)
    ax2.set_ylabel('Accuracy')

#     ax2.tick_params(
#         axis='x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False) # labels along the bottom edge are off

#     ax1.tick_params(
#         axis='x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False) # labels along the bottom edge are off
    
    ax1.set_xlabel('Model Combination')
    
    ax1.grid(False)
    plt.grid(False)

    plt.show()
    
    
class MCDropout(nn.Module):
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
        super(MCDropout, self).__init__()
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
            self.nll = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

        else:
            if class_weights is not None:
                weight_tensor = torch.Tensor(class_weights)
            self.nll = nn.CrossEntropyLoss(weight=weight_tensor)

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
        return [output_tensor]

    
def train_baseline(model, optimizer, loss_fn, train_loader, test_data, test_labels):
    
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
            loss = loss_fn(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader) 
            
        # Validation.
        model.eval()
        model.to('cpu')
        outputs_val = model(test_data)
        val_loss = loss_fn(outputs_val, test_labels)
        
        print(f"Epoch {epoch} \t Validation loss: {val_loss.item():.2f}, Train loss: {train_loss:.2f}")
        
        if val_loss.item() < best_val_loss - 1e-4:
                best_val_loss = val_loss.item()
                patience = 0
                torch.save(model.state_dict(), 'tmp.pth')
        else:
            patience += 1

            if patience == 8:
                model.load_state_dict(torch.load('tmp.pth'))
                return model 
            
            
def accuracy(model, val_data, val_target):
    model_outputs = model(val_data)
    predictions = torch.argmax(model_outputs, dim=-1)
    accuracy = (predictions == val_target).mean()
    print(f"Accuracy: {100 * accuracy:.2f}")
