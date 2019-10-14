# pytorch-msssim

### Differentiable Multi-Scale Structural Similarity (SSIM) index

This small utiliy provides a differentiable MS-SSIM implementation for PyTorch based on Po Hsun Su's implementation of SSIM @ https://github.com/Po-Hsun-Su/pytorch-ssim. I provide asmall changes compared to the method provided by [Jorge Pessoa](https://github.com/jorge-pessoa/pytorch-msssim).
At the moment only a direct method is supported.

## Installation

To install the current version of pytorch_mssim:

1. Clone this repo.
2. Go to the repo directory.
3. Run `python3 -m pip install -e .`

or

1. Clone this repo.
2. Copy "pytorch_msssim" folder in your project.

or

1. `python3 -m pip install git+https://github.com/SpikeAI/pytorch-msssim`


## Example

### Basic usage
```python
import pytorch_msssim
import torch
from torch.autograd import Variable

m = pytorch_msssim.NMSSSIM(val_range=1., normalize=True)

img1 = torch.rand(1, 1, 256, 256)
img2 = torch.rand(1, 1, 256, 256)

print('direct call to MSSSIM:', pytorch_msssim.msssim(img1, img2))
print('Negative MSSSIM as a (derivable) function:', m(img1, img2))


```

### Training

For a detailed example on how to use `msssim` for training, look at the file `test/max_ssim.py`.

We recommend using the flag `normalized=True` when training unstable models using MS-SSIM (for example, Generative Adversarial Networks) as it will guarantee that at the start of the training procedure, the MS-SSIM will not provide NaN results.

## Reference

https://ece.uwaterloo.ca/~z70wang/research/ssim/

https://github.com/Po-Hsun-Su/pytorch-ssim

Thanks to @z70wang for providing the initial SSIM implementation and all the contributors with fixes to this fork and @jorge-pessoa for continuing this work.
