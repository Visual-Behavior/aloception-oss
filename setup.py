from setuptools import setup, find_packages

setup(
    name='aloception',
    author='Visual Behavior',
    version='0.6.0',
    description='Aloception is a set of package for computer vision: aloscene, alodataset, alonet.',
    packages=find_packages(include=["aloscene", "aloscene.*", "alodataset", "alodataset.*", "alonet", "alonet.*"]),
    url='https://visualbehavior.ai/',
    download_url='https://github.com/Visual-Behavior/aloception-oss',
    install_requires=[
        'pycocotools==2.0.2',
        'PyYAML==5.4.1',
        'chardet==4.0.0',
        'idna==2.10',

        'scipy==1.10.0',

        'more_itertools==8.8.0',
        'requests==2.25.1',
        'opencv-python==4.7.0.68',

        'python-dateutil==2.8.2',
        'urllib3==1.26.6',

        'protobuf==4.21.12',
        'wandb==0.13.9',

        'tqdm==4.62.3',
        'captum==0.4.0',

        'setuptools==59.5.0',

        'numpy==1.23.5',

        'pytest==7.2.2',
        'Image==1.5.33'
        ],
    setup_requires=['numpy', 'torch', 'nvidia-pyindex', 'pycuda'],
    license_files=['LICENSE'],
    keywords=['artificial intelligence', 'computer vision'],
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
