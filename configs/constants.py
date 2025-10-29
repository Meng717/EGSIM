import numpy as np

class CoordTransform:
    world2map = np.array([0., 1., 0., 0.,
                          1., 0., 0., 0.,
                          0., 0.,-1., 0.,
                          0., 0., 0., 1.]).reshape(4, 4)
    camera2lidar = np.array([0.,  0., 1., 0.,
                             -1., 0., 0., 0.,
                             0., -1., 0., 0.,
                             0.,  0., 0., 1.]).reshape(4, 4)
    camera2ego = np.array([0., 0., 1., 0., 
                          -1., 0., 0., 0., 
                           0.,-1., 0., 0., 
                           0., 0., 0., 1.]).reshape(4, 4)

class CollisionDetect:
    gaussian_opacity_thresold = 0.5
    gaussian_points_thresold = 50
    lidar_points_thresold = 50