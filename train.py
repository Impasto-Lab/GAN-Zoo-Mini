import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.utils import save_image

from models import dcgan, gan

# Every model file defines Generator and Discriminator, so any module works here
MODELS = {'gan': gan, 'dcgan': dcgan}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gan', choices=MODELS.keys())
    parser.add_argument('-l', '--real-label', type=float, default=1.0,
                        help='Target label of real images for D. 0.9 enables one-sided label smoothing.')
    args = parser.parse_args()

    model = MODELS[args.model]
    out_dir = f'./outputs/{args.model}'
    if args.real_label != 1.0:
        out_dir += f'_real{args.real_label}'  # Do not overwrite the default run

    device = 'cuda'
    batch_size = 256

    # GAN is unsupervised: use train + test (70k images)
    trainset = datasets.MNIST(root='./data', train=True, download=True)
    testset = datasets.MNIST(root='./data', train=False, download=True)

    # Load all images to the GPU once (220 MB), much faster than a per-image DataLoader
    images = torch.cat([trainset.data, testset.data]).to(device)  # uint8 [70000, 28, 28]
    images = (images.float() / 255 - 0.5) / 0.5  # [0, 255] -> [-1, 1], same range as tanh
    images = images.unsqueeze(1)  # [70000, 1, 28, 28]

    # Labels: not used by GAN and DCGAN, needed by cGAN
    labels = torch.cat([trainset.targets, testset.targets]).to(device)  # [70000]

    # Models, optimizers and loss
    generator = model.Generator(128, (1, 28, 28)).to(device)
    discriminator = model.Discriminator((1, 28, 28)).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Fixed noise, so the saved images show how G changes
    fixed_noise = torch.randn(64, 128).to(device)
    os.makedirs(out_dir, exist_ok=True)

    num_batches = len(range(0, len(images), batch_size))

    for epoch in range(100):
        start = time.time()

        # Epoch sums stay on the GPU (.item() at every step makes the CPU wait for the GPU)
        sum_loss_D = sum_loss_G = sum_D_real = sum_D_fake = 0.0

        perm = torch.randperm(len(images), device=device)  # Shuffle
        for k in range(0, len(images), batch_size):
            idx = perm[k:k + batch_size]
            real, real_labels = images[idx], labels[idx]
            noise = torch.randn(real.shape[0], 128).to(device)
            fake = generator(noise)

            # Update D: real -> real_label, fake -> 0. detach() keeps loss_D out of G.
            # real_label < 1 is one-sided label smoothing (Salimans et al., 2016).
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            loss_real = criterion(pred_real, torch.full_like(pred_real, args.real_label))
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

            loss_D = loss_real + loss_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Update G: fake -> 1, i.e. minimize -log D(G(z)) (non-saturating loss).
            # Reuse fake: G is unchanged. Grads this adds to D are cleared next step.
            pred_fake = discriminator(fake)
            loss_G = criterion(pred_fake, torch.ones_like(pred_fake))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            sum_loss_D += loss_D.detach()
            sum_loss_G += loss_G.detach()
            sum_D_real += pred_real.detach().mean()
            sum_D_fake += pred_fake.detach().mean()

        # D(x), D(G(z)): mean D score on real and fake. Both near 0.5: balanced. 1 and 0: D wins.
        print(f'epoch {epoch+1:3d} | '
              f'loss_D {sum_loss_D.item() / num_batches:.3f} | '
              f'loss_G {sum_loss_G.item() / num_batches:.3f} | '
              f'D(x) {sum_D_real.item() / num_batches:.3f} | '
              f'D(G(z)) {sum_D_fake.item() / num_batches:.3f} | '
              f'{time.time() - start:.1f}s')

        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_image((samples + 1) / 2, f'{out_dir}/epoch_{epoch+1:03d}.png', nrow=8)  # [-1, 1] -> [0, 1]

    # Save final weights
    torch.save(generator.state_dict(), f'{out_dir}/generator.pth')
    torch.save(discriminator.state_dict(), f'{out_dir}/discriminator.pth')

if __name__ == '__main__':
    main()
