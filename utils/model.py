from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nflows.transforms import RandomPermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform




def model_definer(num_features, num_iterations, hidden_features, patience, factor, min_lr):

    base_dist = StandardNormal(shape=[num_features])

    transforms = []
    for _ in range(num_iterations):
        transforms.append(RandomPermutation(features=num_features))
        transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,hidden_features=hidden_features))

    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    num_parameters = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print('Num. iterations = {}, Num. hidden_features = {}, Num. trainable parameters = {}'.format(num_iterations,hidden_features,num_parameters))
    optimizer = optim.Adam(flow.parameters())

    flow = flow.to('cuda')

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr)

    return flow, optimizer, scheduler

