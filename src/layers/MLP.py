import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self,
                 dimension_list,
                 prefix="mlp",
                 activation=nn.ReLU(),
                 batch_norm=False,
                 dropout=0.01):

        """
        Create a multi-layer perceptron from the given dimension list.

        :param dimension_list: list of dimensions of the layers
        including input and output layer. Ex. [10, 32, 32, 2]
        will create a MLP with input dimension = 10 and 2 hidden layer
        with dimension 32 and output layer dimension 2.
        :param prefix: a string that will be prepended in the layer names.
        :param activation: non-linear activation after each layer of the mlp
        :param dropout: dropout value
        """

        super(MLP, self).__init__()
        layer_dict = OrderedDict()
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for i in range(1, len(dimension_list)):
            if i != len(dimension_list) - 1:
                layer_dict[prefix + "_drop_" + str(i)] = drop
            layer_dict[prefix + "_linear_" + str(i)] = nn.Linear(dimension_list[i - 1], dimension_list[i])
            if batch_norm:
                layer_dict[prefix + "_bn_" + str(i)] = nn.BatchNorm1d(dimension_list[i])
            if i != len(dimension_list) - 1:
                layer_dict[prefix + "_act_" + str(i)] = activation
        self.mlp = nn.Sequential(layer_dict)

    def forward(self, x):
        return self.mlp(x)

