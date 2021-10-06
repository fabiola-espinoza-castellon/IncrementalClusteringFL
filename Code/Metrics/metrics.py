import torch


def get_accuracy(features, truth, network):
    """

    Parameters
    ----------
    features : torch Tensor
        Input Torch tensor for which we want to get accuracy by 'network'.
    truth : torch Tensor
        Torch tensor corresponding to true labels of 'features'.
    network : class Network
        Network used for inference.

    Returns
    -------
    correct/total : float
        Accuracy in percentage of 'network' for input 'features'.
    """
    pred = get_predictions(features, network)
    correct = (pred == truth).sum().item()
    total = truth.size(0)
    return correct / total


def get_predictions(features, network):
    """

    Parameters
    ----------
    features : torch Tensor
        Input Torch tensor for which output will be predicted by 'network'.
    network : class Network
        Network used for inference.

    Returns
    -------
    pred : torch Tensor
        Torch Tensor of predictions by 'network' of 'features'.
    """
    outputs = network(features)
    _, pred = torch.max(outputs.data, 1)
    return pred
