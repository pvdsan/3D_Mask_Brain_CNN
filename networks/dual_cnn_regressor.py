import torch
import torch.nn as nn
from networks.shared_cnn_1 import RegionFeatureExtractor1
from networks.shared_cnn_2 import RegionFeatureExtractor2
from networks.regressor import Regressor

class DualCNNRegressor(nn.Module):
    def __init__(self):
        super(DualCNNRegressor, self).__init__()
        self.cnn1 = RegionFeatureExtractor1()
        self.cnn2 = RegionFeatureExtractor2()
        self.regressor = Regressor()


    def forward(self, x1, x2, e1, e2):
        
        batch_size = x1.size(0)
        out1 = self.cnn1(x1) # btc_size 174, 10
        out2 = self.cnn2(x2) # btc_size, 39, 10
        out1 = torch.cat((out1, e1), dim=2)  # Shape: [batch_size, 174, 13]
        out2 = torch.cat((out2, e2), dim=2)  # Shape: [batch_size, 39, 13]
        
        # Combine both outputs
        combined = torch.cat((out1, out2), dim=1)  # Shape: [batch_size, 213, 13]
        # Flatten before feeding into the regressor
        combined_flat = combined.view(batch_size, -1)  # Shape: [batch_size, 213*13]
        output = self.regressor(combined_flat)
        return output










