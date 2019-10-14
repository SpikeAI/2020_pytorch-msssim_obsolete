from distutils.core import setup

setup(
  name = 'pytorch_msssim',
  packages = ['pytorch_msssim'], # this must be the same as the name above
  version = '20191014',
  description = 'Using multi-scale structural similarity (MS-SSIM) index for pytorch',
  author = 'Laurent Perrinet',
  author_email = 'laurent.perrinet@univ-amu.fr',
  url = 'https://github.com/SpikeAI/pytorch-msssim', # use the URL to the github repo
  keywords = ['pytorch', 'image-processing', 'deep-learning', 'ms-ssim'], # arbitrary keywords
  classifiers = [],
)
