from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, DataLoader

def main():
    # GAN is unsupervised, so merge train and test sets to get 70k images
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    dataset = ConcatDataset([trainset, testset])

    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    # Sanity check: inspect one batch (labels are not needed for GAN)
    for imgs, _ in dataloader:
        print(imgs.shape, imgs.min().item(), imgs.max().item())
        break

if __name__ == '__main__':
    main()
