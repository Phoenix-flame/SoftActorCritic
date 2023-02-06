import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, n_inputs, n_outputs, output_activation=nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = nn.Linear(in_features=n_inputs, out_features=512)
        self.layer_2 = nn.Linear(in_features=512, out_features=512)
        self.output_layer = nn.Linear(in_features=512, out_features=n_outputs)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = nn.functional.relu(self.layer_1(inpt))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output
