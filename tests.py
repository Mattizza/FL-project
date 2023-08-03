# testare se hanno una miou maggiore le pseudo labels ottenute prima delle transforms o quelle ottenute dopo le transforms
from datasets.idda import IDDADataset
import os
import datasets.ss_transforms as sstr
from models.deeplabv3 import deeplabv3_mobilenetv2
import torch
from torch.utils.data import DataLoader
from utils.stream_metrics import StreamSegMetrics
from tqdm import tqdm
import torch.nn.functional as F
from utils.pseudoLabels import get_image_mask
import numpy as np



class TestPseudolab():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #set the trasnforms
        self.alwaysTransforms = sstr.Compose([
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        #create the dataset
        self.idda_root = 'data/idda'
        with open(os.path.join(self.idda_root,'train.txt'), 'r') as f:
            idda_train_data = f.read().splitlines()
            self.idda_train_dataset_bf_tr = IDDADataset(root=self.idda_root, list_samples=idda_train_data, transform=self.alwaysTransforms,
                                                client_name="idda_test")
        
        #Create the model
        self.model = deeplabv3_mobilenetv2(num_classes=16)
        self.model.cuda()

        #Load the checkpoint
        root1 = 'checkpoints'
        root2 = 'gta'
        path = os.path.join(root1, root2, self.args.checkpoint_to_load)
        checkpoint = torch.load(path)
        print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:"
                  f"\n - epochs executed: {checkpoint['actual_epochs_executed']}"
                  f"\n - Target_eval_miou on {checkpoint['eval_dataset']}: {checkpoint['target_eval_miou']:.2%}\n")
        
        self.model.load_state_dict(checkpoint['model_state'])

        self.metrics = {'pseudo_bf_tr': StreamSegMetrics(n_classes = 16 + 1, name = 'pseudo_bf_tr'), #we add a class that keeps track of the pixels assigned as "dont'care" in the pseudolabels
                        'pseudo_aft_tr': StreamSegMetrics(n_classes = 16 + 1, name = 'pseudo_aft_tr')
                         }

    @staticmethod
    def updateMetricPseudoLbl(metric, outputLogit, labels):
        _ , pseudo_lab = outputLogit.max(dim=1)
        prob = F.softmax(outputLogit, dim=1)
        
        prob = prob[0] #These are necessary since we are evaluating one image at a time
        pseudo_lab = pseudo_lab[0]

        mask = get_image_mask(prob, pseudo_lab)
        pseudo_lab[~mask] = 16 #classe da ignorare

        labels = labels.cpu().numpy() #This is the true label
        prediction = pseudo_lab.cpu().numpy() #This prediction is used as a pseudolabel
        metric.update(labels, np.expand_dims(prediction, axis=0))
    
    
    def _load_metric(self, metric, dataset):
        
        print('Using this transforms:\n')
        print(dataset.transforms)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        metric.reset()

        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(dataloader)):
                images = images.to(self.device) 
                labels = labels.to(self.device)
                outputLogit = self.model(images)['out']
                self.updateMetricPseudoLbl(metric, outputLogit, labels)
    

    def get_metric_pseudo_bf_tr(self):
        print("\nCalculating metrics with pseudo labels before transforms...\n")
        metric = self.metrics['pseudo_bf_tr']
        self._load_metric(metric, self.idda_train_dataset_bf_tr)
        print("\n\nResults:")
        miou = metric.get_results()["Mean IoU"]
        print(f"\tMiou: {miou:.2f}")
        agnosticPxl = metric.confusion_matrix[:,16].sum()

        # totLabeledPxl numb. will be sometimes lower than totImgs * h * w 
        # since we don't count the pxl labeled as 255
        totLabeledPxl = metric.confusion_matrix.sum() 
        agnosticPxlPerc = agnosticPxl/totLabeledPxl
        print(f"\tPerc agnostic pixels: {agnosticPxlPerc:.2f}")
        
        return metric


    def get_metric_pseudo_aft_tr(self, additionalTransforms = None): #transforms is a list of transforms, not a sstr.Compose obj
        
        if additionalTransforms == None:
            print("You should add some additionalTransforms")
            transforms = self.alwaysTransforms
        else:
            baseTransforms = [sstr.ToTensor(), sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transforms = sstr.Compose(additionalTransforms + baseTransforms)
        
        with open(os.path.join(self.idda_root,'train.txt'), 'r') as f:
            idda_train_data = f.read().splitlines()
            dataset = IDDADataset(root=self.idda_root, list_samples=idda_train_data, transform=transforms,
                                                client_name="idda_test")
        
        print("\nCalculating metrics with pseudo labels after transforms...\n")
        metric = self.metrics['pseudo_aft_tr']
        
        self._load_metric(metric, dataset)
        print("\n\nResults:")
        miou = metric.get_results()["Mean IoU"]
        print(f"\tmiou: {miou:.2f}")
        agnosticPxl = metric.confusion_matrix[:,16].sum()

        # totLabeledPxl numb. will be sometimes lower than totImgs * h * w 
        # since we don't count the pxl labeled as 255
        totLabeledPxl = metric.confusion_matrix.sum() 
        agnosticPxlPerc = agnosticPxl/totLabeledPxl
        print(f"\tPerc agnostic pixels: {agnosticPxlPerc:.2f}")

        return metric
    
    def get_results():
        pass