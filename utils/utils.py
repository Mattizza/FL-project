import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import normalize

class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()

#TODO: eliminare se il programma funziona senza
#unNormalize non modifica il tensore originale, restituisce una copia unNormalized
"""def unNormalize(tensorImage, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        copytensorImage = copy.deepcopy(tensorImage)
        for t, m, s in zip(copytensorImage, mean, std):
            t.mul_(s).add_(m)
        return copytensorImage"""

#Non modifica il tensore originale
def denormalize(tensorImage, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    _mean = -np.array(mean) / std
    _std = 1 / np.array(std)
    return normalize(tensorImage, _mean, _std)



class SelfTrainingLoss(nn.Module):
    requires_reduction = False
    # Più la fraction è bassa più sicura è l'assegnazione della classe.
    # Con fraction 0.66 avrai sempre almeno il 66% di pixel assegnati a una classe (non 255)
    def __init__(self, conf_th=0.9, fraction=0.66, ignore_index=255, lambda_selftrain=1, **kwargs):
        super().__init__()
        self.conf_th = conf_th
        self.fraction = fraction
        self.ignore_index = ignore_index
        self.teacher = None
        self.lambda_selftrain = lambda_selftrain

    def set_teacher(self, model):
        #model.eval() #the model is already in eval mode and will stay like this
        self.teacher = model
        #print("is on cuda: ",next(self.teacher.parameters()).is_cuda)
        #print("is in train mode: ", self.teacher.training)

    def get_image_mask(self, prob, pseudo_lab):
        max_prob = prob.detach().clone().max(0)[0] #trova la probabilità massima per ogni pxl
        #la mask restituisce True se la max_prob > soglia 
        mask_prob = max_prob > self.conf_th if 0. < self.conf_th < 1. else torch.zeros(max_prob.size(),
                                                                                       dtype=torch.bool).to(
            max_prob.device)
        #crea una mask con solo False, della stessa dimensione della mask_prob
        mask_topk = torch.zeros(max_prob.size(), dtype=torch.bool).to(max_prob.device)

        if 0. < self.fraction < 1.:
            for c in pseudo_lab.unique():
                mask_c = pseudo_lab == c #mask con True dove la label è c
                max_prob_c = max_prob.clone()
                max_prob_c[~mask_c] = 0 #crea una mask di probabilità che vale 0 dove non c'è la label c
                #Nella riga sotto prende gli indici degli index con prob maggiore nella classe
                _, idx_c = torch.topk(max_prob_c.flatten(), k=int(mask_c.sum() * self.fraction))
                mask_topk_c = torch.zeros_like(max_prob_c.flatten(), dtype=torch.bool)
                mask_topk_c[idx_c] = 1 #mette true gli indici con prob maggiore
                #mask_c passa da avere tutti true a true solamente nei pixel con prob maggiore, scelti prima
                mask_c &= mask_topk_c.unflatten(dim=0, sizes=max_prob_c.size())
                mask_topk |= mask_c #Fa un Or con assegnazione su una mask che all'inizio e solo false,
                                    #e quindi man mano aggiunge true dei pixel corretti
        return mask_prob | mask_topk #La barra verticale è un OR. Affinchè un pixel venga selezionato deve essere stato
                #scelto o dalla mask_prob o dalla mask_top_k

    def get_batch_mask(self, pred, pseudo_lab):
        b, _, _, _ = pred.size() #b è la batch size
        #softmax normalizza le logit. Qundi vengono passate le probabilità e le pseudolabels
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(F.softmax(pred, dim=1), pseudo_lab)], dim=0)
        return mask

    def get_pseudo_lab(self, pred, imgs=None, return_mask_fract=False, model=None):
        teacher = self.teacher if model is None else model #definisce il teacher model
        if teacher is not None:
            with torch.no_grad():
                try:
                    pred = teacher(imgs)['out'] #ricava la predizione del modello
                except:
                    pred = teacher(imgs)
                pseudo_lab = pred.detach().max(1)[1] #restituisce la label con massima logit
        else:
            pseudo_lab = pred.detach().max(1)[1]
        mask = self.get_batch_mask(pred, pseudo_lab)
        pseudo_lab[~mask] = self.ignore_index #dove nella mask c'è il valore False mette l'indice da ignorare
        if return_mask_fract:
            return pseudo_lab, F.softmax(pred, dim=1), mask.sum() / mask.numel()
        return pseudo_lab

    def forward(self, pred, imgs=None): #queste pred sono quelle dello student model
        pseudo_lab = self.get_pseudo_lab(pred, imgs) #calcola le pseudo labels
        #calcola la loss tra le pred dello student model e le pseudo_labels
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean() * self.lambda_selftrain #lambda attribuisce un peso a questa loss
