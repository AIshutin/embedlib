import setuptools

setuptools.setup(
    name="embedlib",
    version="0.1",
    description="PyTorch drop-in embedders and tokenizers",
    packages=['embedlib'],
    install_requires=['pytorch-transformers',],
)
