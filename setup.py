from setuptools import setup, find_packages

setup(
    name='aloception',
    author='Visual Behavior',
    version='0.3.0',
    description='Aloception is a set of package for computer vision: aloscene, alodataset, alonet.',
    packages=find_packages(include=['aloscene', 'alodataset', 'alonet']),
    url='https://visualbehavior.ai/',
    download_url='https://github.com/Visual-Behavior/aloception-oss',
    install_requires=[
        'matplotlib==3.5.3',
        'more-itertools==8.8.0', # required for alodataset waymo
        'numpy==1.23.2',
        'onnx==1.12.0',
        'onnx_graphsurgeon==0.0.1.dev5',
        'onnxsim==0.4.8',
        'opencv-python==4.5.3.56'
        'Pillow==9.2.0',
        'pycocotools==2.0.2',    # required for alodataset coco
        'pytorch_lightning==1.4.1',
        'pytorch_quantization==0.0.1.dev5',
        'Requests==2.28.1',
        'scipy==1.4.1',          # required for alonet/detr/matcher
        'setuptools==63.4.1',
        'tensorflow==2.10.0',    # required for alodataset/prepare/waymo_converter
        'tensorrt==0.0.1.dev5',
        'torchvision==0.13.1',
        'tqdm==4.62.3',
        'ts==0.5.1',
        'wandb==0.12.2',
        'waymo_open_dataset==1.0.1'],
    setup_requires=['torch', 'nvidia-pyindex', 'pycuda'],
    license_files=['LICENSE'],
    keywords=['artificial intelligence', 'computer vision'],
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
