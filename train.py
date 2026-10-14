from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1] to match tanh output
    ])

    # GAN is unsupervised, so merge train and test sets to get 70k images
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataset = ConcatDataset([trainset, testset])

    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    # Sanity check: inspect one batch (labels are not needed for GAN)
    for imgs, _ in dataloader:
        print(imgs.shape, imgs.min().item(), imgs.max().item())
        break

if __name__ == '__main__':
    main()
