import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

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

    # Models, optimizers and loss
    generator = Generator(128, 784).to(device)
    discriminator = Discriminator(784).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Same noise every time, so the saved images show how G changes over training
    fixed_noise = torch.randn(64, 128).to(device)
    os.makedirs('./outputs', exist_ok=True)

    num_batches = len(range(0, len(data), batch_size))

    for epoch in range(100):
        start = time.time()

        # Sums over the epoch, kept on the GPU (calling .item() at every step makes the GPU wait)
        sum_loss_D = sum_loss_G = sum_D_real = sum_D_fake = 0.0

        perm = torch.randperm(len(data), device=device)  # New random order every epoch
        for k in range(0, len(data), batch_size):
            # One batch of real images and the same number of fake images
            real = data[perm[k:k + batch_size]]
            noise = torch.randn(real.shape[0], 128).to(device)
            fake = generator(noise)

            # Update D: label real as 1 and fake as 0.
            # detach() cuts the graph at fake, so loss_D does not flow back into G.
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            loss_real = criterion(pred_real, torch.ones_like(pred_real))
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

            loss_D = loss_real + loss_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Update G: label fake as 1, i.e. minimize -log D(G(z)) instead of log(1 - D(G(z))).
            # The paper suggests this because log(1 - D(G(z))) gives almost no gradient early on.
            # fake is reused: G has not changed since it was made, so its graph is still valid.
            # loss_G.backward() also adds grads to D, but opt_D.zero_grad() clears them next step.
            pred_fake = discriminator(fake)
            loss_G = criterion(pred_fake, torch.ones_like(pred_fake))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            sum_loss_D += loss_D.detach()
            sum_loss_G += loss_G.detach()
            sum_D_real += pred_real.detach().mean()
            sum_D_fake += pred_fake.detach().mean()

        # D(x): mean score of D on real images. D(G(z)): mean score of D on fake images.
        # Both near 0.5 means G and D are balanced. D(x) -> 1 and D(G(z)) -> 0 means D wins.
        print(f'epoch {epoch+1:3d} | '
              f'loss_D {sum_loss_D.item() / num_batches:.3f} | '
              f'loss_G {sum_loss_G.item() / num_batches:.3f} | '
              f'D(x) {sum_D_real.item() / num_batches:.3f} | '
              f'D(G(z)) {sum_D_fake.item() / num_batches:.3f} | '
              f'{time.time() - start:.1f}s')

        # Save a sample grid every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                samples = generator(fixed_noise).view(-1, 1, 28, 28)
            save_image((samples + 1) / 2, f'./outputs/epoch_{epoch+1:03d}.png', nrow=8)  # [-1, 1] -> [0, 1]

    # Save the final weights
    torch.save(generator.state_dict(), f'./outputs/generator.pth')
    torch.save(discriminator.state_dict(), f'./outputs/discriminator.pth')

if __name__ == '__main__':
    main()
