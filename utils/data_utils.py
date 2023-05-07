import numpy as np


def idda_16_cmap(n=256):
    color = [[128, 64, 128],
             [244, 35, 232],
             [70, 70, 70],
             [102, 102, 156],
             [190, 153, 153],
             [153, 153, 153],
             [250, 170, 30],
             [220, 220, 0],
             [107, 142, 35],
             [152, 251, 152],
             [70, 130, 180],
             [220, 20, 60],
             [255, 0, 0],
             [0, 0, 142],
             [0, 0, 230],
             [119, 11, 32],
             [0, 0, 0]]
    cmap = np.zeros((n, 3), dtype='uint8')
    for i, co in enumerate(color):
        cmap[i] = co
    return cmap.astype(np.uint8)


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]
