import torch
from models.basic.basic_class import BasicSimpleRenderer
from lib.models.street_gaussian_renderer import StreetGaussianRenderer

renderer = StreetGaussianRenderer()  

class StreetGaussianSimpleRenderer(BasicSimpleRenderer):
    def razterization(self, camera, gaussians):
        with torch.no_grad():                  
            torch.cuda.synchronize()
            result = renderer.render_all(camera, gaussians)            
            torch.cuda.synchronize()
        return result['rgb'].permute(1,2,0).detach().cpu().numpy(), None