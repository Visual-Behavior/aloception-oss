Getting started
--------------------------------------------------

About Aloception
===========================

.. image:: ../images/aloception-oss.jpg
  :width: 400
  :alt: Alternative text

Aloception is a set of packages for computer vision built on top of popular deep learning libraries:
`pytorch <https://pytorch.org/>`_  and  `pytorch lightning <https://www.pytorchlightning.ai/>`_ .

| **Aloscene** extends the use of
  `tensors <https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_tensor.html>`_ with **Augmented Tensors**
  and **Spatial Augmented Tensors**. The latter are designed to facilitate the use of computer vision data
  (such as frames, 2d boxes, 3d boxes, optical flow, disparity, camera parameters...).


| **Alodataset** implements ready-to-use datasets for computer vision with the help of **aloscene** and **augmented tensors** to make it easier to transform and display your vision data.


| **Alonet** integrates several promising computer vision architectures. You can use it for research purposes or to quickly finetune or deploy your model using TensorRT. Alonet is mainly built on top  of `pytorch lightning <https://www.pytorchlightning.ai/>`_ with the help of
  **aloscene** and **alodataset**.

.. note::
    One can use **aloscene** independently than the two other packages to handle computer vision data, or to improve its
    training pipelines with **augmented tensors**.


Install
===========================

Aloception's packages are built on top of multiple libraries. Most of them are listed in the **requirements.txt**
file::

    pip install -r requirements.txt

Once the other packages are installed, you still need to install pytorch based on your hardware and environment
configuration. Please, ref to the `pytorch website <https://pytorch.org/>`_  for this install.


Other optional installation
===========================

Deformable DETR:  build Multi-scale Deformable Attention ops::

    cd alonet/deformable_detr/ops
    ./make.sh
    python test.py # should yield True

Please check gcc compatibility with your CUDA Toolkit Version. For example: `CUDA Toolkit 11.4.0 <https://docs.nvidia.com/cuda/archive/11.4.0/cuda-installation-guide-linux/index.html>`_
Or `other versions of CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`_.

Exporting to tensorRT: TensorRT toolkit need to be installed on your system. Once done, the following pip packages
are required::

    pip instal onnx
    pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
    pip install tensorrt
