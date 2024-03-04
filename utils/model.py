"""
This file contains the definition of the model and related functions for training and optimization.

The model_definer function defines a flow-based model using the nflows library. It takes the following parameters:
- num_features: The number of input features for the model.
- num_iterations: The number of base distributions to generate the flow.
- hidden_features: The number of hidden features in the MaskedAffineAutoregressiveTransform.
- patience: The number of epochs with no improvement after which the learning rate will be reduced.
- factor: The factor by which the learning rate will be reduced.
- min_lr: The minimum learning rate.
- initial_lr: The initial learning rate.

The function returns the following objects:
- flow: The flow-based model.
- optimizer: The optimizer used for training the model.
- scheduler: The learning rate scheduler.

The model is defined using a base distribution (StandardNormal) and a series of transformations. The transformations include
RandomPermutation and MaskedAffineAutoregressiveTransform. The CompositeTransform combines all the transformations.

The model is trained using the Adam optimizer and the learning rate is reduced using the ReduceLROnPlateau scheduler.

The function also prints the number of iterations, number of hidden features, and number of trainable parameters in the model.
"""

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nflows.transforms import RandomPermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform

def model_definer(num_features, num_iterations, hidden_features, patience, factor, min_lr, initial_lr):
    """
    Define a flow-based model using the nflows library.

    Args:
        num_features : The number of input features for the model.
        num_iterations : The number of iterations to perform in the flow.
        hidden_features : The number of hidden features in the MaskedAffineAutoregressiveTransform.
        patience : The number of epochs with no improvement after which the learning rate will be reduced.
        factor : The factor by which the learning rate will be reduced.
        min_lr : The minimum learning rate.
        initial_lr : The initial learning rate.

    Returns:
        flow : The flow-based model.
        optimizer : The optimizer used for training the model.
        scheduler : The learning rate scheduler.
    """
    base_dist = StandardNormal(shape=[num_features])

    transforms = []
    for _ in range(num_iterations):
        transforms.append(RandomPermutation(features=num_features))
        transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,hidden_features=hidden_features))

    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    num_parameters = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print('Num. iterations = {}, Num. hidden_features = {}, Num. trainable parameters = {}'.format(num_iterations,hidden_features,num_parameters))
    optimizer = optim.Adam(flow.parameters(), lr=initial_lr)

    flow = flow.to('cuda')

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr)

    return flow, optimizer, scheduler

