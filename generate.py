import argparse
import os

import torch
from torchvision.utils import save_image

from models import cgan, dcgan, gan

MODELS = {'gan': gan, 'dcgan': dcgan, 'cgan': cgan}  # Same as train.py
Z_DIM = 128  # Must match train.py
IMG_SHAPE = (1, 28, 28)


def main():
    parser = argparse.ArgumentParser(description='Generate images with a trained generator.')
    parser.add_argument('-m', '--model', type=str, default='gan', choices=MODELS.keys())
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Path of the generator weights. Default: outputs/<model>/generator.pth.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path of the output image. Default: outputs/<model>/generated.png.')
    parser.add_argument('-n', '--num', type=int, default=None,
                        help='Number of images. Default: 64 (8 x 8), or 100 (10 x 10) for cgan.')
    parser.add_argument('-d', '--digit', type=int, default=None, choices=range(10),
                        help='cgan only: generate only this digit. Default: column i is digit i.')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed. The same seed gives the same images.')
    args = parser.parse_args()

    conditional = args.model == 'cgan'
    if args.digit is not None and not conditional:
        parser.error('--digit needs --model cgan')

    checkpoint = args.checkpoint or f'./outputs/{args.model}/generator.pth'
    output = args.output or f'./outputs/{args.model}/generated.png'
    num = args.num or (100 if conditional else 64)
    nrow = 10 if conditional and args.digit is None else 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    generator = MODELS[args.model].Generator(Z_DIM, IMG_SHAPE).to(device)
    generator.load_state_dict(torch.load(checkpoint, map_location=device))
    generator.eval()  # BatchNorm uses its running statistics, not the statistics of this batch

    noise = torch.randn(num, Z_DIM, device=device)
    with torch.no_grad():
        if not conditional:
            samples = generator(noise)
        else:
            if args.digit is None:
                labels = torch.arange(num, device=device) % 10  # Column i is digit i
            else:
                labels = torch.full((num,), args.digit, device=device)
            samples = generator(noise, labels)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    save_image((samples + 1) / 2, output, nrow=nrow)  # [-1, 1] -> [0, 1]
    print(f'Saved {num} images to {output}')


if __name__ == '__main__':
    main()
