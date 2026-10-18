import math

import torch
import torch.nn as nn

# MLP: z [B, in_dim] -> image [B, C, H, W] in [-1, 1]
class Generator(nn.Module):
    def __init__(self, in_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, math.prod(img_shape))
        self.act = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x.view(-1, *self.img_shape)  # [B, C*H*W] -> [B, C, H, W]

# MLP: image [B, C, H, W] -> P(real) [B, 1]
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.fc1 = nn.Linear(math.prod(img_shape), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.drop = nn.Dropout(0.2)  # As in the paper
        self.act = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(1)  # [B, C, H, W] -> [B, C*H*W]
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        return self.sig(x)

if __name__ == '__main__':
    # Shape check with small sizes
    device = 'cuda'
    noise = torch.randn(2, 16).to(device)

    generator = Generator(16, (1, 8, 8)).to(device)
    discriminator = Discriminator((1, 8, 8)).to(device)

    fake = generator(noise)
    logit = discriminator(fake)

    print(fake.shape, logit.shape)
