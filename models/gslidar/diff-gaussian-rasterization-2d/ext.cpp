/*
 * Copyright (C) 2025, Fudan Zhang Vision Group
 * All rights reserved.
 * 
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}