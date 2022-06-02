from setuptools import setup, find_packages

setup(
  name = 'enformer-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.5.2',
  license='MIT',
  description = 'Enformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/enformer-pytorch',
  keywords = [
    'artificial intelligence',
    'transformer',
    'gene-expression'
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'torchmetrics',
    'polars',
    'pyfaidx',
    'pyyaml',
    'transformers',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
