import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN: z [B, in_dim] -> image [B, C, H, W] in [-1, 1]
# Upsample + conv replaces the paper's ConvTranspose2d
class Generator(nn.Module):
    def __init__(self, in_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.fc = nn.Linear(in_dim, 7 * 7 * 64)  # 7 = 28 / 2 / 2
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 1, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x).view(x.size(0), 64, self.img_shape[1]//4, self.img_shape[2]//4)
        x = self.act(self.bn0(x))
        x = F.interpolate(x, mode='nearest', scale_factor=2)  # 7 -> 14
        x = self.act(self.bn1(self.conv1(x)))
        x = F.interpolate(x, mode='nearest', scale_factor=2)  # 14 -> 28
        x = self.conv2(x)  # No BN on G's output layer (paper)
        return self.tanh(x)

# CNN: image [B, C, H, W] -> P(real) [B, 1]
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(7 * 7 * 64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.conv1(x))  # 28 -> 14, no BN on D's input layer (paper)
        x = self.act(self.bn2(self.conv2(x)))  # 14 -> 7
        x = x.flatten(1)
        x = self.fc(x)
        return self.sig(x)

if __name__ == '__main__':
    # Shape check with small sizes
    device = 'cuda'
    noise = torch.randn(2, 16).to(device)

    generator = Generator(16, (1, 28, 28)).to(device)
    discriminator = Discriminator((1, 28, 28)).to(device)

    fake = generator(noise)
    logit = discriminator(fake)

    print(fake.shape, logit.shape)
