from setuptools import setup, find_packages
from __version__ import __version__


setup(
    name="aloception",
    author="Visual Behavior",
    version=__version__,
    description="Aloception is a set of package for computer vision: aloscene, alodataset, alonet.",
    packages=find_packages(include=["aloscene", "aloscene.*", "alodataset", "alodataset.*", "alonet", "alonet.*"]),
    url="https://visualbehavior.ai/",
    download_url="https://github.com/Visual-Behavior/aloception-oss",
    install_requires=[
        "PyYAML==6.0.2",
        "chardet==4.0.0",
        "idna==2.10",
        "scipy==1.10.0",
        "more_itertools==8.8.0",
        "requests==2.25.1",
        "opencv-python==4.7.0.68",
        "python-dateutil==2.8.2",
        "urllib3==1.26.6",
        "protobuf==4.21.12",
        "wandb==0.17.9",
        "tqdm==4.62.3",
        "captum==0.4.0",
        "setuptools==59.5.0",
        "tensorboard>=2.13.0",
    ],
    setup_requires=["numpy", "torch", "nvidia-pyindex", "pycuda"],
    license_files=["LICENSE"],
    keywords=["artificial intelligence", "computer vision"],
    classifiers=["Programming Language :: Python", "Topic :: Scientific/Engineering :: Artificial Intelligence"],
)
