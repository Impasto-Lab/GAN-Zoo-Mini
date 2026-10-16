from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset
import torch.optim as optim
import torch.nn as nn
import torch
import os

from models.gan import Generator, Discriminator


def main():
    device = 'cuda'
    batch_size = 256

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1] to match tanh output
    ])

    # GAN is unsupervised, so merge train and test sets to get 70k images
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataset = ConcatDataset([trainset, testset])

    # Load the whole dataset to the GPU once (MNIST is only 220 MB).
    # This is faster than a DataLoader, which transforms each image on the CPU in every epoch.
    data = torch.stack([img for img, _ in dataset]).flatten(1).to(device)  # [70000, 784]

    # Create models
    generator = Generator(128, 784).to(device)
    discriminator = Discriminator(784).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Same noise every time, so the saved images show how G changes over training
    fixed_noise = torch.randn(64, 128).to(device)
    os.makedirs('./outputs', exist_ok=True)

    for epoch in range(100):
        perm = torch.randperm(len(data), device=device)  # Shuffle
        for k in range(0, len(data), batch_size):
            imgs = data[perm[k:k + batch_size]]

            noise = torch.randn(imgs.shape[0], 128).to(device)
            fake = generator(noise)

            pred_real = discriminator(imgs)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))

            pred_fake = discriminator(fake.detach())
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

            loss_D = loss_real + loss_fake
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            fake = generator(noise)
            pred_fake = discriminator(fake)
            loss_G = criterion(pred_fake, torch.ones_like(pred_fake))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        # Save a sample grid every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                samples = generator(fixed_noise).view(-1, 1, 28, 28)
            save_image((samples + 1) / 2, f'./outputs/epoch_{epoch+1:03d}.png', nrow=8)  # [-1, 1] -> [0, 1]

    torch.save(generator.state_dict(), f'./outputs/generator.pth')
    torch.save(discriminator.state_dict(), f'./outputs/discriminator.pth')

if __name__ == '__main__':
    main()
