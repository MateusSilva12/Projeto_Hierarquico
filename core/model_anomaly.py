import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyDetectionCNN(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, image_size=64):
        super(AnomalyDetectionCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
        )
        
        # Latent space
        self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, 4, 2, 1),
            nn.Tanh(),
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), 128, 4, 4)
        return self.decoder(z)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction

class SimpleAnomalyDetector(nn.Module):
    """Modelo mais simples para datasets menores"""
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleAnomalyDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
