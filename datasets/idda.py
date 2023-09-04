import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import datasets.ss_transforms as sstr
import torch
import torch.nn.functional as F
from utils.utils import denormalize



class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]


class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: [str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        self.root = root

        self.teacherModel = None
        self.pseudoLBeforeTransforms = False

    @staticmethod
    def get_mapping():
        """
        mappa le labels che non usiamo a 255 e scala le altre.
        Esempio:
        mappa = [255, 0, 1, 2]
        originale = [3, 2,   0, 2, 1]
        mappato =   [2, 1, 255, 1, 0]
        """
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        """
        :return: torch.tensor()
        """
        imagePath = os.path.join(self.root, 'images' , self.list_samples[index]+'.jpg')
        labelPath = os.path.join(self.root, 'labels' ,self.list_samples[index]+'.png')

        image = Image.open(imagePath)

        if  self.pseudoLBeforeTransforms:
            transformBase = sstr.Compose([sstr.ToTensor(), sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            tensorImage = transformBase(image)
            tensorImage = tensorImage.to("cuda:0")
            tensorImage = torch.unsqueeze(tensorImage, 0)
            #print("tensor image type",type(tensorImage))
            #print("tensor image shape", tensorImage.shape)
            self.teacherModel.eval()
            with torch.no_grad():
                outputLogit = self.teacherModel(tensorImage)['out']

                pseudo_lab = outputLogit.detach().max(1)[1]

                prob = F.softmax(outputLogit, dim=1)
                
                prob = prob[0]
                pseudo_lab = pseudo_lab[0]

              
            mask = self.get_image_mask(prob, pseudo_lab)
            pseudo_lab[~mask] = 255 #classe da ignorare
            print("Our pseudo label Shape:", pseudo_lab.shape)
            print()
            print("Our pseudo label unique:", pseudo_lab.unique())
            print()
            print(pseudo_lab)
            print("end")

            numpy_array = pseudo_lab.cpu().numpy()
            label = Image.fromarray(numpy_array.astype(np.uint8))#it is a PIL Image
            print(torch.from_numpy(np.array(label, dtype=np.uint8)))
            print(pseudo_lab.device)
            print(torch.from_numpy(np.array(label, dtype=np.uint8)).device)
            print("\nPrimo check:")
            print((pseudo_lab.cpu() == torch.from_numpy(np.array(label, dtype=np.uint8))).unique())

        else:
            label = Image.open(labelPath)

        if self.transform is not None:
            image, label = self.transform(image, label)
        
        if self.target_transform is not None and self.pseudoLBeforeTransforms == False:
            label = self.target_transform(label)
        
        #Da rimuovere if sotto (solo per debugging)
        if  self.pseudoLBeforeTransforms:    
            print("\nSecondo check:")
            print((label == pseudo_lab.cpu()).unique())

        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
    
    def showImgAndLable(self, index):
        image, label = self.__getitem__(index)
        #denormalize to show
        fig, axs = plt.subplots(2)
        #fig.set_figheight(8.275)
        #fig.set_figwidth(15)
        fig.suptitle('Transformed and Segmentated Image')
        #fig.tight_layout()
        for i in range(2):
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        axs[0].imshow(denormalize(image).permute(1, 2, 0))
        axs[1].imshow(label.numpy())

    def set_teacherModel(self, teacherModel):
        self.teacherModel = teacherModel
        self.pseudoLBeforeTransforms = True
    
    def get_image_mask(self, prob, pseudo_lab, conf_th = 0.9, fraction = 0.66):
        max_prob = prob.detach().clone().max(0)[0] #trova la probabilità massima per ogni pxl
        #la mask restituisce True se la max_prob > soglia 
        mask_prob = max_prob > conf_th if 0. < conf_th < 1. else torch.zeros(max_prob.size(),
                                                                                       dtype=torch.bool).to(
            max_prob.device)
        #crea una mask con solo False, della stessa dimensione della mask_prob
        mask_topk = torch.zeros(max_prob.size(), dtype=torch.bool).to(max_prob.device)

        if 0. < fraction < 1.:
            for c in pseudo_lab.unique():
                mask_c = pseudo_lab == c #mask con True dove la label è c
                max_prob_c = max_prob.clone()
                max_prob_c[~mask_c] = 0 #crea una mask di probabilità che vale 0 dove non c'è la label c
                #Nella riga sotto prende gli indici degli index con prob maggiore nella classe
                _, idx_c = torch.topk(max_prob_c.flatten(), k=int(mask_c.sum() * fraction))
                mask_topk_c = torch.zeros_like(max_prob_c.flatten(), dtype=torch.bool)
                mask_topk_c[idx_c] = 1 #mette true gli indici con prob maggiore
                #mask_c passa da avere tutti true a true solamente nei pixel con prob maggiore, scelti prima
                mask_c &= mask_topk_c.unflatten(dim=0, sizes=max_prob_c.size())
                mask_topk |= mask_c #Fa un Or con assegnazione su una mask che all'inizio e solo false,
                                    #e quindi man mano aggiunge true dei pixel corretti
        return mask_prob | mask_topk #La barra verticale è un OR. Affinchè un pixel venga selezionato deve essere stato
                #scelto o dalla mask_prob o dalla mask_top_k
    
    def set_transforms(self, composeTransform):
        self.transforms = composeTransform
