import torch.nn as nn
## Total parameters : 279,306

class RegionFeatureExtractor1(nn.Module):
    def __init__(self, num_classes=10):
        super(RegionFeatureExtractor1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=0, bias = False),  # Output: (32, 25, 25, 25)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # Output: (32, 12, 12, 12)
            
            nn.Conv3d(32, 64, kernel_size=3, padding=0,bias = False),  # Output: (64, 10, 10, 10)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),  # Output: (64, 5, 5, 5)   # 
            
            
            nn.Conv3d(64, num_classes, kernel_size=5, padding = 0)
        )

    def forward(self, x):
        batch_size = x.shape[0] #8
        num_regions = x.shape[1] # 174
        x = x.view(batch_size * num_regions, 1, 27, 27, 27)
        x = self.features(x)
        x = x.view(batch_size, num_regions, -1) # [8, 174, 10]
        return x
