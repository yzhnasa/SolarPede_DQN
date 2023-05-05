import torch
import torch.nn as nn
import torch.nn.functional as F


# MAX_LASER_CURRENT = 6.0
# MAX_LASER_FREQUENCY = 6000


# Q_VALUE_SIZE = 8
class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_unites):
        super(Net, self).__init__()
        self.state_size = state_size
        self.hidden_unites = hidden_unites
        self.action_size = action_size
        self.fully_connected_layer_1 = nn.Linear(state_size, hidden_unites)
        # self.batch_normal_layer_1 = nn.BatchNorm1d(num_features=hidden_unites)
        self.fully_connected_layer_2 = nn.Linear(hidden_unites, hidden_unites)
        # self.batch_normal_layer_2 = nn.BatchNorm1d(num_features=hidden_unites)
        self.fully_connected_layer_3 = nn.Linear(hidden_unites, hidden_unites)
        # self.batch_normal_layer_3 = nn.BatchNorm1d(num_features=hidden_unites)
        self.fully_connected_layer_4 = nn.Linear(hidden_unites, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fully_connected_layer_1.weight.data.normal_(0, 0.1)
        self.fully_connected_layer_2.weight.data.normal_(0, 0.1)
        self.fully_connected_layer_3.weight.data.normal_(0, 0.1)
        self.fully_connected_layer_4.weight.data.normal_(0, 0.1)

    def forward(self, state):
        # print("state dim: ", state.dim())
        # print("state size: ", state.size())
        x = self.fully_connected_layer_1(state)
        # x = self.batch_normal_layer_1(x)
        x = torch.tanh(x)

        x = self.fully_connected_layer_2(x)
        # x = self.batch_normal_layer_2(x)
        x = torch.tanh(x)

        x = self.fully_connected_layer_3(x)
        # x = self.batch_normal_layer_3(x)
        x = torch.tanh(x)

        x = self.fully_connected_layer_4(x)
        action = torch.sigmoid(x)
        action = action.squeeze()
        # print("action dim: ", action.dim())
        # print("action size: ", action.size())
        return action

