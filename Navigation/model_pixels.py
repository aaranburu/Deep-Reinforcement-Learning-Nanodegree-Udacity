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
        
        # 4 dimension input, 32 output channels/feature maps
        # 8x8 square convolution kernel
        ## output size = (W-F)/S +1 = (84-3)/4 +1 = 21
        # the output Tensor for one image, will have the dimensions: (action_size, 21, 21)
        #self.conv1 = nn.Conv2d(state_size, 32, 8, stride=4)
        self.conv1 = nn.Conv2d(state_size, state_size*action_size, 3, stride=4)
        
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2) 
        
        # 32 dimension input, 64 output channels/feature maps
        # 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (10-3)/1 +1 = 8
        # the output Tensor for one image, will have the dimensions: (4*action_size, 8, 8)
        #self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2 = nn.Conv2d(state_size*action_size, 2*action_size*state_size, 3, stride=1)
        
        # 64 dimension input, 64 output channels/feature maps
        # 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (9-2)/1 +1 = 8
        # the output Tensor for one image, will have the dimensions: (64, 7, 7)
        #self.conv3 = nn.Conv2d(64, 64, 2, stride=1)
        
        # 64 outputs * the 5*5 filtered map size as input, 512 outputs
        #self.fc1 = nn.Linear(64*7*7, 512)
        self.fc1 = nn.Linear(2*action_size*state_size*4*4, 32)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # 64 outputs * the 5*5 filtered map size as input, 512 outputs
        self.fc2 = nn.Linear(32, action_size)
        
        '''
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        '''

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        #print(state.size(0), state.size(1), state.size(2), state.size(3))
        x = self.pool(F.relu(self.conv1(state.float())))
        #print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size(0), x.size(1), x.size(2), x.size(3))
        #x = F.relu(self.conv3(x))
        #print(x.size(0), x.size(1), x.size(2), x.size(3))
        
        x = x.view(x.size(0), -1)
        #print(x.size(0), x.size(1))
        x = F.relu(self.fc1(x))
        #print(x.size(0), x.size(1))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        #print(x.size(0), x.size(1))
        
        return x
        
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        '''
        
        
        
