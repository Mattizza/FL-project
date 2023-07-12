import copy
from collections import OrderedDict
from client import Client
from torch.utils.data import DataLoader
import numpy as np
import torch
import wandb
from torch import optim, nn
from utils.utils import HardNegativeMining, MeanReduction
from inspect import signature
from torch.optim import lr_scheduler
#from utils import definePath
import os

class ServerGTA:
    def __init__(self, args, source_dataset, test_clients, model, metrics):
        self.args = args

        self.source_dataset = source_dataset
        self.test_clients = test_clients #lista con tre elementi (idda_test, idda_same_dom, idda_diff_dom)

        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict()) #da eliminare
        #per setting centralized
        self.params = None #da eliminare
        self.train_loader = DataLoader(self.source_dataset, batch_size=self.args.bs, shuffle=True, drop_last=True)
        self.mious = {'idda_test':[], "test_same_dom":[], "test_diff_dom":[]}

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = None
        self.scheduler = None

        self.t = 1 #how many epochs between the evaluations
        self.max_eval_miou = 0.0
        self.actual_epochs_to_max_eval_miou = 0
        
    def _get_outputs(self, images):
        
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        
        if self.args.model == 'resnet18':
            return self.model(images)
        
    def set_optimizer(self, config):
        """
            Creates an optimizier with the parameters contained in config. Also discard the useless settings.
        """
        opt_config = config.get('optimizer')
        opt_name = opt_config.get('name')
        opt_method = getattr(optim, opt_name)
        opt_signature  = set(signature(opt_method).parameters.keys())
        valid_params_k = opt_signature.intersection(set(opt_config))
        valid_params_dict = {key: opt_config[key] for key in valid_params_k}
        
        self.optimizer = opt_method(self.model.parameters(), **valid_params_dict)
        if self.args.framework == 'centralized':
            print(self.optimizer)
        

    def set_scheduler(self, config):
        """
            Creates a scheduler with the parameters contained in config. Also discard the useless settings.
        """
        sch_config = config.get('scheduler')
        sch_name = sch_config.get('name')
        if sch_name != 'None':
            sch_method = getattr(lr_scheduler, sch_name)
            sch_signature  = set(signature(sch_method).parameters.keys())
            valid_params_k = sch_signature.intersection(set(sch_config))
            valid_params_dict = {key: sch_config[key] for key in valid_params_k}
            self.scheduler = sch_method(self.optimizer, **valid_params_dict)
        
        if self.args.framework == 'centralized':
            if sch_name != 'None':
                print('Scheduler:\n',type(self.scheduler),"\n", self.scheduler.state_dict())
            else:
                print("No scheduler")


    def  create_opt_sch(self, config: dict):
        """
            Simply call the method to create the optimizer and the scheduler
        """
        #il file config che riceve deve essere un dizionario con chiavi esterne "optimizer" e "scheduler"
        #se usi config = wand.config viene automaticamente fatto in questo modo

        self.set_optimizer(config)
        self.set_scheduler(config)    



    def run_epoch(self, cur_epoch, n_steps: int) -> None:
        '''
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level.
            -) cur_epoch: current epoch of training;
        '''
        cumu_loss = 0
        print('Epoch', cur_epoch + 1)
        for cur_step, (images, labels) in enumerate(self.train_loader):
   
            images = images.to(self.device, dtype = torch.float32) 
            labels = labels.to(self.device, dtype = torch.long)
            outputs = self._get_outputs(images)
            
            loss = self.reduction(self.criterion(outputs,labels), labels)
            cumu_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # === Wandb ===
            if self.args.wandb != None and self.args.framework == 'centralized':
                wandb.log({"batch loss": loss.item()})

            # === Printing === 
            # We keep track of the loss. Notice we are storing the loss for
            # each mini-batch.
            #wandb.log({"loss": loss.mean()})
            
            # We are considering 10 batch at a time. TO DO: define a way to handle different values.
            # We are considering n_steps batch at a time.
            self.n_total_steps = len(self.train_loader)
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
                self.n_10th_steps.append(self.count)
                print(f'epoch {cur_epoch + 1} / {self.args.num_epochs}, step {cur_step + 1} / {self.n_total_steps}, loss = {loss.mean():.3f}')
        
        return cumu_loss / len(self.train_loader)

    def train(self, n_steps=10):
        '''
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        '''

        self.model.train()

        # Training loop. We initialize some empty lists because we need to store the information about the statistics computed
        # on the mini-batches.
        self.n_total_steps = len(self.train_loader)
        self.mean_loss = []
        self.mean_std  = []
        self.n_10th_steps = []
        self.n_epoch_steps = [self.n_total_steps]
        self.count = 0

       
        num_train_samples = len(self.source_dataset)

        # We iterate over the epochs.
        for epoch in range(self.args.num_epochs):
            
            # wandb
            if self.args.wandb != None:
                    wandb.log({"lr": self.optimizer.param_groups[0]['lr']})

            avg_loss = self.run_epoch(epoch, n_steps)

            #if there is a scheduler do a step at each epoch
            if self.scheduler != None:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                    print("Reduce On Plateau Scheduler, lr:", self.optimizer.param_groups[0]['lr'])
                else:
                    self.scheduler.step()
                    print("Altro Scheduler, lr:", self.optimizer.param_groups[0]['lr'])
                
            # wandb
            if self.args.wandb != None:
                wandb.log({"loss": avg_loss, "epoch": epoch})
            actual_epochs_executed = epoch +1

            #Check the miou on idda_train every t epochs
            if actual_epochs_executed % self.t == 0 or actual_epochs_executed == (self.args.num_epochs):
                eval_miou = self.eval_train()
                self.model.train()

                # ==== Saving the model ====
                #compare current eval_miou with the max_eval_miou
                if eval_miou > self.max_eval_miou and self.args.name_checkpoint_to_save != None:
                    self.max_eval_miou = eval_miou
                    self.save_checkpoint(eval_miou, actual_epochs_executed)



            # Here we are simply computing how many steps do we need to complete an epoch.
            self.n_epoch_steps.append(self.n_epoch_steps[0] * (epoch + 1))
        
        avg_loss_last_epoch = avg_loss
        return avg_loss_last_epoch
    
    def save_checkpoint(self, eval_miou, actual_epochs_executed):
        
        print(f"=> Saving checkpoint. Target_eval_miou: {eval_miou:.2%}\n")

        checkpoint = {"model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "target_eval_miou": eval_miou,
                    "actual_epochs_executed": actual_epochs_executed}

        root = 'checkpoints'
        customPath = self.args.name_checkpoint_to_save
        path = os.path.join(root, customPath)
        torch.save(checkpoint, path)
        
        print('Server saved checkpoint at ', path)
        
    
    def load_model_opt_sch(self):
        root = 'checkpoints'
        path = os.path.join(root, self.args.checkpoint_to_load)
        checkpoint = torch.load(path)
        print(f"\n=> Loading the trained model:\n - epochs executed: {checkpoint['actual_epochs_executed']}\n - Target_eval_miou: {checkpoint['target_eval_miou']}")
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.max_eval_miou = checkpoint['target_eval_miou']


    def eval_train(self):
        """
        This method handles the evaluation on the train clients.
        Reset the metrics computed at the previous round, load the model on each
        train client, test the model on the client dataset, update the
        StreamMetric (SM) object. (Note: there is just a single SM obj for all the
        training clients).
        """

        metric = self.metrics['idda_test']
        metric.reset()
        print(f"Evaluating on the train set of Idda")
        client = self.test_clients[0]
        self.load_server_model_on_client(client)
        client.test(metric)

        miou = metric.get_results()['Mean IoU']
        
        # wandb
        if self.args.wandb != None:
            wandb.log({'idda_train_miou':miou})

        print(f'Mean IoU: {miou:.2%} \n')
        self.mious['idda_test'].append(miou)
        
        return miou
     
        
  

    def test(self):
        """
        This method handles the test on the test clients.
        Load the server model on each test client, reset the previously computed
        metrics, test the model on the test client's dataset
        """
        for client in self.test_clients[1:]: #skip the first client since already tested
            print(f"Testing client {client.name}...")
            self.load_server_model_on_client(client)
            metric = self.metrics[client.name]
            metric.reset()
            client.test(metric)
            miou = metric.get_results()['Mean IoU']
            if self.args.wandb != None:
                wandb.log({client.name : miou})
            print(f'Mean IoU: {miou:.2%}')
            self.mious[client.name].append(miou)


    def load_server_model_on_client(self, client):
        client.model = self.model #<- con questa passi proprio il modello
        #client.model.load_state_dict(copy.deepcopy(self.model.state_dict())) #<- con questa passi i parametri del modello


    