import torch
import torch.nn as nn

class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(0.25) 
        self.relu = nn.ReLU()
        
        sample_input = torch.zeros(1, 1, 28, 28)
        conv_output = self._forward_conv(sample_input)
        self.fc1 = nn.Linear(conv_output.numel(), 512)  
        self.fc2 = nn.Linear(512, 10)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
