import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.drop = nn.Dropout(0.2)
        self.act = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        return self.sig(x)

if __name__ == '__main__':
    device = 'cuda'
    noise = torch.randn(1, 16).to(device)

    generator = Generator(16, 128).to(device)
    discriminator = Discriminator(128).to(device)

    fake = generator(noise)
    logit = discriminator(fake)

    print(logit.shape)