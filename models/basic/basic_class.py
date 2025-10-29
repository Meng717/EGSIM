
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import torch
import os
import glob
from torch import Tensor
from models.basic.utils.graphics_utils import getTransformMatrix

__all__ = ['BasicSimpleLoader', 'BasicSimpleRenderer', 
           'BasicLoader', 'BasicRenderer', 'BasicSensor']

class BasicSimpleLoader(ABC):
    def __init__(self):
        super().__init__()
        self._model_path = None

        self._scene = None
        self._cameras = None
        self._gaussians = None
    
    def set(self):
        pass
    
    def load(self):
        self._scene = self.initScene()
        self._cameras = self.getCameras(self._scene)
        self._gaussians = self.getGaussians(self._scene)
    
    @property
    def cameras(self):
        return self._cameras
    @property
    def gaussians(self):
        return self._gaussians
    
    @abstractmethod
    def initScene(self):
        pass
    
    @abstractmethod
    def getCameras(self, scene):
        pass

    @abstractmethod
    def getCameraExtrinsic(self, camera):
        pass

    @abstractmethod
    def getCameraIntrinsic(self, camera):
        pass

    @abstractmethod
    def getCameraWidth(self, camera):
        pass

    @abstractmethod
    def getCameraHeight(self, camera):
        pass

    @abstractmethod
    def getCameraFar(self, camera):
        pass

    @abstractmethod
    def getCameraNear(self, camera):
        pass

    @abstractmethod
    def setCameraExtrinsic(self, camera, transform):
        pass

    @abstractmethod
    def getGaussians(self, scene):
        pass

class BasicSimpleRenderer(ABC):
    def __init__(self):
        super().__init__()
    
    def set(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def razterization(self, camera, gaussians):
        pass

class BasicLoader(ABC):
    def __init__(self):
        super().__init__()
        self._data_path = None
        self._model_path = None
    @property
    def data_path(self):
        return self._data_path
    @property
    def model_path(self):
        return self._model_path
    def set(self, data_path, model_path):
        self._data_path = data_path
        self._model_path = model_path
    @abstractmethod
    def load(*args, **kwargs):
        pass

class BasicSensorLoader(BasicLoader):
    def __init__(self):
        super().__init__()
        self._frame_num = None
        self._ego_poses = None
    @property
    def frame_num(self):
        return self._frame_num
    @property
    def ego_poses(self):
        return self._ego_poses
    def set(self, data_path, model_path):
        super().set(data_path, model_path)
        self._ego_poses = [np.loadtxt(pose) for pose in 
                           sorted(glob.glob(os.path.join(self.data_path, "pose", "*.txt")))]
        self._frame_num = len(self._ego_poses)
    @abstractmethod
    def load(self):
        pass

class BasicModelLoader(BasicLoader):
    pass


class BasicRenderer(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def set(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def getSettings(self) -> dict:
        pass
    
    @abstractmethod
    def razterization(self, *args, **kwargs):
        pass

@dataclass
class BasicSensorInfo:
    idx: int
    rotation: np.ndarray      # Camera to world
    position: np.ndarray      # Camera to world
    _init_rotation: np.ndarray = field(init=False, repr=False)
    _init_position: np.ndarray = field(init=False, repr=False)
    transform: np.ndarray = np.eye(4)
    scale_factor: float = 1.0
    device: torch.device = torch.device('cuda')
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self._init_rotation = self.rotation
        self._init_position = self.position

    @property
    def c2w(self) -> Tensor:
        return torch.tensor(getTransformMatrix(self.rotation, self.position), 
                            device=self.device, dtype=self.dtype)
    @property
    def w2c(self) -> Tensor:
        return self.c2w.inverse()
    @property
    def transformed_rotation(self) -> np.ndarray:
        return self.transformed_c2w.cpu().numpy()[:3, :3]
    @property
    def transformed_position(self) -> np.ndarray:
        return self.transformed_c2w.cpu().numpy()[:3, 3]
    @property
    def transformed_c2w(self) -> Tensor:
        result = np.diag(np.array([1 / self.scale_factor] * 3 + [1])) @ self.transform @ self.c2w.cpu().numpy()
        result[:3, 3] *= self.scale_factor
        return torch.tensor(result, dtype=self.dtype, device=self.device)
    @property
    def transformed_w2c(self) -> Tensor:
        return self.transformed_c2w.inverse()
    @property
    def init_rotation(self) -> np.ndarray:
        return self._init_rotation
    @property
    def init_position(self) -> np.ndarray:
        return self._init_position
    @property
    def init_c2w(self) -> Tensor:
        return torch.tensor(getTransformMatrix(self.init_rotation, self.init_position), 
                            dtype=self.dtype, device=self.device)
    @property
    def init_w2c(self) -> Tensor:
        return self.init_c2w.inverse()
    @property
    def init_transformed_rotation(self) -> np.ndarray:
        return self.init_transformed_c2w.cpu().numpy()[:3, :3]
    @property
    def init_transformed_position(self) -> np.ndarray:
        return self.init_transformed_c2w.cpu().numpy()[:3, 3]
    @property
    def init_transformed_c2w(self) -> Tensor:
        result = np.diag(np.array([1 / self.scale_factor] * 3 + [1])) @ self.transform @ self.init_c2w.cpu().numpy()
        result[:3, 3] *= self.scale_factor
        return torch.tensor(result, dtype=self.dtype, device=self.device)
    @property
    def init_transformed_w2c(self) -> Tensor:
        return self.init_transformed_c2w.inverse()