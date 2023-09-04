import torch

def get_image_mask(prob, pseudo_lab, conf_th = 0.9, fraction = 0.66):
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