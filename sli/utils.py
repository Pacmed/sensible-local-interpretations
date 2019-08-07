"""Utilities for both training and experiments."""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score


"""Empty dictionary with performance metrics to use while Bayesian training."""
PERFORMANCE_DICT = {
    'loss': 0,
    'complexity_cost': 0,
    'nll': 0,
    'log_variational_posterior': 0,
    'log_prior': 0,
    'accuracy': 0,
    'auc': 0,
    'f1_score': 0,
    'average_precision': 0,
    'id_uncertainty': 0,
    'ood_uncertainty': 0,
    'uncertainty_ratio': 0,
    'uncertainty_difference': 0
}


def write_performance_scalars(writer,
                              log_dir: str,
                              step: int,
                              performance_dict: dict):
    """Write training loss performances to TensorBoardX SummaryWriter.

    Parameters
    ----------
    writer: SummaryWriter
        Dictionary with losses.
    log_dir: str
        Name of the logs (logs / test_logs etc.)
    step: int
        Current step: epoch * batch_idx
    performance_dict: dict
        Dictionary with loss scalars to write.

    """

    for k in performance_dict:
        path = os.path.join(log_dir, k)
        if performance_dict[k] != 0:
            writer.add_scalar(path, round(performance_dict[k], 4), step)


def cross_entropy(y_true: np.array, y_score: np.array, n_labels: int = None,
                  reduction: str = None, epsilon=1e-9) -> np.array:
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
    epsilon: float
        Numerical stability.

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

        ce = -(y_true * np.log2(y_score + epsilon) + (1 - y_true) * np.log2(1 - y_score + epsilon))

    else:
        # One-hot encode y_true
        y_true = np.eye(n_labels)[y_true]

        if y_true.shape != y_score.shape:
            raise ValueError("y_true shape not equal to y_score shape")

        ce = -np.sum(y_true * np.log2(y_score + epsilon), axis=-1)

    # Apply reduction.
    if reduction == 'sum':
        return ce.sum()
    if reduction == 'mean':
        return ce.mean()

    return ce


def get_performance_vs_uncertainty(y_true: np.array,
                                      y_pred: np.array,
                                      y_unc: np.array,
                                      y_axis_label: str,
                                      performance_fn: callable = cross_entropy,
                                      performance_fn_args: dict = None):
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
    first_index = max(64,
                      np.argwhere(sorted_labels != sorted_labels[0])[0][0])
    performances = []
    percentages = []

    for i in range(first_index + 1, len(sorted_uncertainties)):
        selected_labels = sorted_labels[:i]
        selected_predictions = sorted_predictions[:i]

        percentages.append(100 * len(selected_predictions) / len(y_pred))

        performances.append(performance_fn(selected_labels, selected_predictions,
                                           **performance_fn_args))

    return percentages, performances
    


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
    
    percentages, performances = get_performance_vs_uncertainty(y_true, y_pred, y_unc, y_axis_label, performance_fn, performance_fn_args)
    fig = plt.figure()
    sns.lineplot(percentages, performances)
    plt.xlabel('% of Uncertain Data')
    plt.ylabel(y_axis_label)
    return fig


def get_performance_metrics(y_true: np.array,
                            y_score: np.array,
                            accuracy_cutoff: float = 0.5):
    """Get current performance metrics.

    Parameters
    ----------
    y_true: np.array
    y_score: np.array
    accuracy_cutoff: float
        When to predict positive predictions in binary case.

    """

    if len(y_score.shape) == 1:
        y_pred = (y_score > accuracy_cutoff).astype(int)

        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_score)

        f1 = f1_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_score)

    else:
        y_pred = y_score.argmax(LAST_DIM)
        auc = 0
        f1 = 0
        ap = 0

    acc = accuracy_score(y_true, y_pred)

    return auc, f1, ap, acc

