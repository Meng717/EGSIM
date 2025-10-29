from dataclasses import dataclass
from models.basic.basic_class import BasicSensorInfo
from models.basic.utils.graphics_utils import getProjectionMatrixCenterShift, \
                                              getProjectionMatrix, \
                                              focal2fov, fov2focal
import torch
from torch import Tensor

@dataclass
class PinholeCamInfo(BasicSensorInfo):
    fx: float = ...
    fy: float = ...
    cx: float = ...
    cy: float = ...
    width: int = ...
    height: int = ...
    image_path: str = None
    near: float = 0.01
    far: float = 100.0

    @property
    def projection_matrix(self, center_shift=True) -> Tensor:
        if center_shift:
            return torch.tensor(getProjectionMatrixCenterShift(
                                self.near, self.far, self.cx, self.cy, 
                                self.fx, self.fy, 
                                self.width, self.height), 
                                device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(getProjectionMatrix(
                                self.near, self.far, self.fovx, self.fovy), 
                                device=self.device, dtype=self.dtype)
    @property
    def full_proj_transform(self) -> Tensor:
        return self.projection_matrix @ self.w2c
    @property
    def fovx(self):
        return focal2fov(self.fx, self.width)
    @fovx.setter
    def fovx(self, value):
        self.fx = fov2focal(value, self.width)
    @property
    def fovy(self):
        return fov2focal(self.fy, self.height)
    @fovy.setter
    def fovy(self, value):
        self.fy = fov2focal(value, self.height)
    @property
    def intr_mat(self) -> Tensor:
        mat = torch.zeros((3, 3), device=self.device, dtype=self.dtype)
        mat[0, 0] = self.fx
        mat[0, 2] = self.cx
        mat[1, 1] = self.fy
        mat[1, 2] = self.cy
        mat[2, 2] = 1
        return mat
    @intr_mat.setter
    def intr_mat(self, value):
        assert type(value) == Tensor
        self.fx = value[0, 0]
        self.cx = value[0, 2]
        self.fy = value[1, 1]
        self.cy = value[1, 2]
    @property
    def transformed_full_proj_transform(self) -> Tensor:
        return self.projection_matrix @ self.transformed_w2c
