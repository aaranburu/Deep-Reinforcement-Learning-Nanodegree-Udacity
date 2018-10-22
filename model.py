import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        '''
        # 4 dimension input, 32 output channels/feature maps
        # 8x8 square convolution kernel
        ## output size = (W-F)/S +1 = (84-8)/4 +1 = 20
        # the output Tensor for one image, will have the dimensions: (32, 20, 20)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        
        # 32 dimension input, 64 output channels/feature maps
        # 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (20-4)/2 +1 = 9
        # the output Tensor for one image, will have the dimensions: (64, 9, 9)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        
        # 64 dimension input, 64 output channels/feature maps
        # 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (9-3)/1 +1 = 7
        # the output Tensor for one image, will have the dimensions: (64, 7, 7)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1)
        
        # 64 outputs * the 5*5 filtered map size as input, 512 outputs
        self.fc1 = nn.Linear(64*7*7, 512)
        
        # 64 outputs * the 5*5 filtered map size as input, 512 outputs
        self.fc1 = nn.Linear(512, action_size)
        '''
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        '''
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.fc1(state))
        
        action = F.relu(self.fc2(state))
        
        return action
        '''
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
        
        
