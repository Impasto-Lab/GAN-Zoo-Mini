# GAN Zoo Mini

This is a collection of PyTorch reproductions of classic GAN papers. I write each model line by line to learn how it works.

The first stage covers three papers:

1. **GAN**: a generator and a discriminator (both MLP) play a minimax game. The generator learns to map noise `z` to images.
2. **cGAN**: the generator and the discriminator both get the class label `y`. This lets you select the digit to generate.
3. **DCGAN**: convolutional layers replace the MLP. The paper also gives rules for stable training (strided convolutions, BatchNorm, LeakyReLU, no pooling).

All models use MNIST (train and test sets together, 70,000 images of size 28 &times; 28).

## Models

| Model | Paper | Venue | Status |
|:---:|:---:|:---:|:---:|
| GAN | [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | NeurIPS 2014 | In progress |
| cGAN | [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) | arXiv 2014 | Planned |
| DCGAN | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) | ICLR 2016 | Planned |

## Examples

<!-- TODO: Add one column per model with a grid of generated samples (for example 8 x 8). Put the images in assets/examples/. -->

*Results will be added after training.*

## Installation

1. Create and activate a conda environment:

   ```bash
   conda create -n gan python=3.12
   ```

   ```bash
   conda activate gan
   ```

2. Install PyTorch and torchvision. Select the command for your CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/). This project uses PyTorch `2.14.1` and torchvision `0.29.1`.

<!-- TODO: Add requirements.txt and a `pip install -r requirements.txt` step. -->

## Usage

<!-- TODO: Add the command for generate.py after the CLI is done. -->

*Will be added after `generate.py` is done.*

## Training

The training script downloads MNIST to `data/` on the first run.

```bash
# Train from scratch
python train.py
```

<!-- TODO: Add the flag to select the model, the resume command and a table of the main parameters. -->

## Project Structure

```text
GAN-Zoo-Mini/
├── data/            # MNIST (downloaded on the first run)
├── models/          # One file per model (planned: gan.py, cgan.py, dcgan.py)
├── generate.py      # Sample images from a trained checkpoint
├── train.py         # Training loop
├── LICENSE
└── README.md
```

## Reference

```bibtex
@inproceedings{goodfellow2014gan,
  title     = {Generative Adversarial Nets},
  author    = {Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2014}
}

@article{mirza2014cgan,
  title   = {Conditional Generative Adversarial Nets},
  author  = {Mirza, Mehdi and Osindero, Simon},
  journal = {arXiv preprint arXiv:1411.1784},
  year    = {2014}
}

@inproceedings{radford2016dcgan,
  title     = {Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  author    = {Radford, Alec and Metz, Luke and Chintala, Soumith},
  booktitle = {International Conference on Learning Representations},
  year      = {2016}
}
```

## License

This project uses the [MIT License](LICENSE).
