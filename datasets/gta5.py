import os
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image
import torch


class GTA5(VisionDataset):

    labels2train  = {'idda': {
            1: 13, # ego_vehicle : vehicle
            7: 0, # road
            8: 1, # sidewalk
            11: 2, # building
            12: 3, # wall
            13: 4, # fence
            17: 5, # pole
            18: 5, # poleGroup: pole
            19: 6, # traffic light
            20: 7, # traffic sign
            21: 8, # vegetation
            22: 9, # terrain
            23: 10, # sky
            24: 11, #person
            25: 12, # rider
            26: 13, # car : vehicle
            27: 13, # truck : vehicle
            28: 13, # bus : vehicle
            32: 14, # motorcycle
            33: 15, # bicycle
            }}

    def __init__(self, client_name: str = None, transform=None, test_transform=None, test = False, target_dataset=None):
        """
        Params:
        * test: if True apply only test_transform, if False apply only transform
        * target_dataset: {'idda'}
        """
        self.root = 'data/gta5'
        super().__init__(self.root, transform=transform, target_transform=None)
        self.test = test
        self.labels2train = GTA5.labels2train[target_dataset]
        self.test_transform = test_transform
        self.target_transform = self.__map_labels()
        self.client_name = client_name
        self.return_original = False

        self.apply_only_fda = False #just for debugging

        #!
        self.style_tf_fn = None #style_transfer_function
        
        with open(os.path.join(self.root, 'train.txt'), 'r') as f:
            self.list_samples = f.read().splitlines()

        #train_path = os.path.join(self.root, 'train.txt')
        #self.list_samples = [ids for ids in open(train_path)]
    
    def __getitem__(self, index: int):
        """
        :return: torch.tensor()
        """
        transform = self.transform if not self.test else self.test_transform

        imagePath = os.path.join(self.root, 'images' , self.list_samples[index])
        labelPath = os.path.join(self.root, 'labels' ,self.list_samples[index])

        image = Image.open(imagePath)
        label = Image.open(labelPath)

        if self.return_original == False:

            if self.style_tf_fn is not None:
                image = self.style_tf_fn(image)
            
            if not self.apply_only_fda:
                if transform is not None:
                    image, label = transform(image, label)
                
                if self.target_transform is not None:
                    label = self.target_transform(label)

        return image, label

    def __map_labels(self):
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in self.labels2train.items():
            mapping[k] = v
        return lambda x: torch.from_numpy(mapping[x])
    
    def __len__(self):
        return len(self.list_samples)
    
    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn
    
    #def add_fda(self):
    #    #TODO: aggiungi davanti a tutte  le transforms il cambio di stile
    #    self.transform.insert(0, fda)








    