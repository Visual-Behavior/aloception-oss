#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

ALONET_ROOT=$1
cd $ALONET_ROOT/deformable_detr/ops
if [ -d "./build" ] 
then
    rm -r ./build
    echo "build directory exists. build directory cleaned." 
fi
python setup.py build install
python test.py
cd -