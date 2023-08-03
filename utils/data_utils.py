import numpy as np


def idda_16_cmap(n=256):
    color = [[128, 64, 128], #Road
             [244, 35, 232], #SideWalk
             [70, 70, 70], #Building
             [102, 102, 156], #Wall
             [190, 153, 153], #Fence
             [153, 153, 153], #Pole
             [250, 170, 30], #Traffic lights
             [220, 220, 0], #Traffic sign
             [107, 142, 35], #Vegetation
             [152, 251, 152], #Terrain
             [70, 130, 180], #Sky
             [220, 20, 60], #Pedestrian
             [255, 0, 0], #Rider
             [0, 0, 142], #Vehicle
             [0, 0, 230], #Motorcycle
             [119, 11, 32], #Bicycle
             [0, 0, 0]] #DoNotCare
    cmap = np.zeros((n, 3), dtype='uint8')
    for i, co in enumerate(color):
        cmap[i] = co
    return cmap.astype(np.uint8)


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]
