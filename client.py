import copy
import torch
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt

from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from utils.data_utils import idda_16_cmap, Label2Color
from utils.utils import HardNegativeMining, MeanReduction, unNormalize

from matplotlib.patches import Rectangle
from collections import defaultdict
from inspect import signature


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, n_steps: int) -> None:
        '''
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level.
            -) cur_epoch: current epoch of training;
            -) optimizer: optimizer used for the local training.
        '''

        print('epoch', cur_epoch + 1)
        for cur_step, (images, labels) in enumerate(self.train_loader):
                
            # Total steps needed to complete an epoch. Computed as:
            # 
            #           self.n_total_steps = floor(len(self.dataset) / self.args.bs).
            # 
            # Example: len(self.dataset) = 600, self.args.bs = 16, self.n_total_steps = 37
            self.n_total_steps = len(self.train_loader)

            images = images.to(self.device, dtype = torch.float32) 
            labels = labels.to(self.device, dtype = torch.long)
            outputs = self._get_outputs(images)
            
            loss = self.reduction(self.criterion(outputs,labels), labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # We keep track of the loss. Notice we are storing the loss for
            # each mini-batch.
            # WARNING: to be decided how to handle this across various clients.
            wandb.log({"loss": loss.mean()})
            
            # We are considering n_steps batch at a time.
            if (cur_step + 1) % n_steps == 0 or cur_step + 1 == self.n_total_steps:

                # We store the information about the steps using self.count. This is needed if we want to plot the learning curve.
                if (cur_step + 1) % n_steps == 0:
                    self.count += n_steps
                else:
                    self.count += (cur_step + 1) % n_steps  # We need to consider the special case in which we have a number of batches
                                                            # different from the steps we fixed (e.g. each 10 steps, but only 7 left).
                
                # We store the values of the mean loss and of the std each n_steps mini-batches (e.g. mean loss of the 10th mini-batch).
                self.mean_loss.append(loss.mean().cpu().detach().numpy())
                self.mean_std.append(loss.std().cpu().detach().numpy())
                
                # We store the information related to the step in which we computed the loss of the n_steps-th mini-batch. This will
                # be of use when plotting the learning curve.
                # WARNING: do we still need this? Do we want to plot the value for different clients?
                self.n_10th_steps.append(self.count)
                print(f'epoch {cur_epoch + 1} / {self.args.num_epochs}, step {cur_step + 1} / {self.n_total_steps}, loss = {loss.mean():.3f} ± {(loss.std() / np.sqrt(self.args.bs)):.3f}')

    
    def train(self, config = None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        
        
        with(wandb.init(config = config,
                        project = self.project, tags = self.tags, notes = self.notes)):
             
            self.model.train()
            config = wandb.config
                    
            # Freeze parameters so we don't backprop through them.
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print('Params freezed')

            # We extract the names, we'll need them later to extract the methods as well. Notice that now
            # we are acting on the 'config' object and its attributes.
            self.opt, self.sch = config.optimizer['name'], config.scheduler['name']
            self.opt_method, self.sch_method = getattr(optim, self.opt), getattr(lr_scheduler, self.sch)
            
            # We filter only the arguments we are interested in. We do this by extracting the signature of
            # the choosen functions and by filtering only the instanciated values thar correspond to it.
            
            # Optimizer signature
            opt_signature = set(signature(getattr(optim, self.opt)).parameters.keys())      
            
            # Parameters that belong to the signature only. We build a dictionary that will be of use later.
            filtered_opt = opt_signature.intersection(set(config.optimizer['settings']))                                  
            dic_opt = config.optimizer['settings']
            opt_we_want = {key: dic_opt[key] for key in filtered_opt}

            # We do the same we just did for the optimizer.
            sch_signature = set(signature(getattr(lr_scheduler, self.sch)).parameters.keys())
            filtered_sch = sch_signature.intersection(set(config.scheduler['settings']))
            dic_sch = config.scheduler['settings']
            sch_we_want = {key: dic_sch[key] for key in filtered_sch}


            # We build the effective optimizer and scheduler. We need first to create fake dictionaries to pass as argument.
            dummy_dict = {'params': self.model.classifier.parameters()}
            dummy_dict.update(opt_we_want)
            self.optimizer = self.opt_method([dummy_dict])
            dummy_dict = {'optimizer': self.optimizer}      # WARNING: WHY 'optimizer' IF WE ARE REFERRING TO THE SCHEDULER? CHECK
            dummy_dict.update(sch_we_want)
            self.scheduler = self.sch_method(**dummy_dict)


            # Training loop. We initialize some empty lists because we need to store the information about the statistics computed
            # on the mini-batches.
            # WARNING: to be decided if it is worth it keep this. Do we need learning curves?
            self.n_total_steps = len(self.train_loader)
            self.mean_loss = []
            self.mean_std  = []
            self.n_10th_steps = []
            self.n_epoch_steps = [self.n_total_steps]
            self.count = 0

            # We iterate over the epochs.
            for epoch in range(self.args.num_epochs):

                self.run_epoch(epoch, self.n_steps)
                self.scheduler.step()
                    
                # Here we are simply computing how many steps do we need to complete an epoch.
                self.n_epoch_steps.append(self.n_epoch_steps[0] * (epoch + 1))
                    
            
            print('Training finished!')
            torch.save(self.model.classifier.state_dict(), 'modelliSalvati/checkpoint.pth')
            print('Model saved!')

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        # TODO: missing code here!
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                # TODO: missing code here!
                raise NotImplementedError
                self.update_metric(metric, outputs, labels)
