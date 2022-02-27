import torch
import torch.nn as nn



def _tie_or_clone_weights(first_module, second_module):
    """ Tie or clone module weights depending of weither we are using TorchScript or not
    """
    if self.config.torchscript:
        first_module.weight = nn.Parameter(second_module.weight.clone())
    else:
        first_module.weight = second_module.weight

    if hasattr(first_module, 'bias') and first_module.bias is not None:
        first_module.bias.data = torch.nn.functional.pad(
            first_module.bias.data,
            (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
            'constant',
            0
        )
