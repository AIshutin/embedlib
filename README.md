# embedlib

Embedlib is a library similar to [sentence-transformers](https://github.com/UKPLab/sentence-transformers). The main advantages of embedlib are:
- the support of Russian datasets and models
- multi GPU support
- half-precision mode
- usage of `sacred` experiment manager
- benchmark system (including memory usage and inference time)

## CLI help
For training use `scripts/train.py`. Check out `sacred` for cli usage.
