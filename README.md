# Complex Steerable Pyramid in PyTorch

This is a PyTorch implementation of the Complex Steerable Pyramid described in [Portilla and Simoncelli (IJCV, 2000)](http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Portilla99). 

It uses PyTorch's efficient spectral decomposition layers `torch.fft` and `torch.ifft`. Just like a normal convolution layer, the complex steerable pyramid expects a batch of images of shape `[N,C,H,W]` with current support only for grayscale images (`C=1`). It returns a `list` structure containing the low-pass, high-pass and intermediate levels of the pyramid for each image in the batch (as `torch.Tensor`). Computing the steerable pyramid is significantly faster on the GPU as can be observed from the runtime benchmark below. 

<a href="/assets/coeff.png"><img src="/assets/coeff.png" width="700px" ></a>

## Usage

In addition to the PyTorch implementation defined in `SCFpyr_PyTorch` the original SciPy version is also included in `SCFpyr` for completeness and comparison. As the GPU implementation highly benefits from parallelization, the `cwt` and `power` methods expect signal batches of shape `[N,H,W]` containing a batch of `N` images of shape `HxW`.

```python
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils

# Load batch of images [N,1,H,W]
im_batch_numpy = utils.load_image_batch(...)
im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)

# Requires PyTorch with MKL when setting to 'cpu' 
device = torch.device('cuda:0')

# Initialize Complex Steerbale Pyramid
pyr = SCFpyr_PyTorch(height=5, nbands=4, scale_factor=2, device=device)

# Decompose entire batch of images 
coeff = pyr.build(im_batch_torch)

# Reconstruct batch of images again
im_batch_reconstructed = pyr.reconstruct(coeff)

```

## Benchmark

Performing parallel the CSP decomposition on the GPU using PyTorch results in a significant speed-up. Increasing the batch size will give faster runtimes. The plot below shows a comprison between the `scipy` versus `torch` implementation as function of the batch size `N` and input signal length. These results were obtained on a powerful Linux desktop with NVIDIA Titan X GPU.

<a href="/assets/runtime_benchmark.pdf"><img src="/assets/runtime_benchmark.png" width="700px" ></a>

## Installation

Clone and install:

```sh
git clone https://github.com/tomrunia/PyTorchSteerablePyramid.git
cd PyTorchSteerablePyramid
pip install -r requirements.txt
python setup.py install
```

## Requirements

- Python 2.7 or 3.6 (other versions might also work)
- Numpy (developed with 1.15.4)
- Scipy (developed with 1.1.0)
- PyTorch >= 0.4.0 (developed with 1.0.0; see note below)

The steerable pyramid depends utilizes `torch.fft` and `torch.ifft` to perform operations in the spectral domain. At the moment, PyTorch only implements these operations for the GPU or with the MKL back-end on the CPU. Therefore, if you want to run the code on the CPU you might need to compile PyTorch from source with MKL enabled. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

## References

- [J. Portilla and E.P. Simoncelli, Complex Steerable Pyramid (IJCV, 2000)](http://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf)
- [The Steerable Pyramid](http://www.cns.nyu.edu/~eero/steerpyr/)
- [Official implementation: matPyrTools](http://www.cns.nyu.edu/~lcv/software.php)
- [perceptual repository by Dzung Nguyen](https://github.com/andreydung/Steerable-filter)

## License

MIT License

Copyright (c) 2018 Tom Runia (tomrunia@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.