import copy
from models.basic.basic_class import BasicSimpleLoader
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.config.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.camera_utils import Camera

class StreetGaussianSimpleLoader(BasicSimpleLoader):
    def initScene(self):
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        return scene
    
    def getCameras(self, scene: Scene):
        return {0: scene.getTrainCameras()}

    def getCameraExtrinsic(self, camera: Camera):
        return camera.get_extrinsic()
    
    def getCameraIntrinsic(self, camera: Camera):
        return camera.get_intrinsic()
    
    def getCameraWidth(self, camera: Camera):
        return camera.image_width

    def getCameraHeight(self, camera: Camera):
        return camera.image_height

    def getCameraFar(self, camera: Camera):
        return camera.zfar

    def getCameraNear(self, camera: Camera):
        return camera.znear

    def setCameraExtrinsic(self, camera: Camera, transform):
        transformed_camera = copy.deepcopy(camera)
        transformed_camera.set_extrinsic(transform)
        return transformed_camera

    def getGaussians(self, scene: Scene):
        return scene.gaussians

