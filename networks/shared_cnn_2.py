import torch.nn as nn
#import torch

# Total parameters: 279,306

class RegionFeatureExtractor2(nn.Module):
    def __init__(self, num_classes=10):
        super(RegionFeatureExtractor2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=0, stride = 2, bias = False),  # Output: (32, 54, 54, 54)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # Output: (32, 18, 18, 18)
    
            nn.Conv3d(32, 64, kernel_size=3, padding=0, bias = False),  # Output: (64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),  # Output: (64, 5, 5, 5)
    
            nn.Conv3d(64, num_classes, kernel_size=5, padding = 0)
            
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_regions = x.shape[1]
        x = x.view(batch_size * num_regions, 1, 56,56,56)
        x = self.features(x)
        x = x.view(batch_size, num_regions, -1) # [8, 39, 10]
        return x
