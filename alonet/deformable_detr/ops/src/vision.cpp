/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_deform_attn.h"
#include <torch/script.h>

// static auto registry = torch::RegisterOperators("alonet::ms_deform_attn", &ms_deform_attn_forward);

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
//   m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
// }

TORCH_LIBRARY(alonet_custom, m){
  m.def("ms_deform_attn_forward", ms_deform_attn_forward);
  m.def("ms_deform_attn_backward", ms_deform_attn_backward);
}