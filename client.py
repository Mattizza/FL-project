import copy
import torch
import numpy as np

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.data_utils import idda_16_cmap, Label2Color

from utils.utils import HardNegativeMining, MeanReduction, denormalize, SelfTrainingLoss
import os
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
from matplotlib.patches import Rectangle
import wandb
from inspect import signature
from tqdm import tqdm
import torch.nn.functional as F
import scipy

from style_extractor import StyleExtractor


class Client:

    def __init__(self, args, train_dataset, test_dataset, model, test_client=False, isTarget = False):
        
        
        self.args = args

        #Datasets and loaders
        self.train_dataset = train_dataset if not test_client else None 
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        self.name = self.test_dataset.client_name
        
        #Models
        self.model = model
        #! da rimuovere se si passa dal main 
        self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #choose loss
        if args.self_train == 'true':
            self.self_train_loss = SelfTrainingLoss()
            #self.self_train_loss.set_teacher(self.model) Rimosso perchè viene settato dal server
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
            self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        self.isTarget = isTarget

        #Task 3.2
        if self.isTarget:
            self.avg_style = None
            self.win_sizes = None
            self.style_extractor = StyleExtractor(self.test_dataset)

        #Task 5
        self.cluster_id = None
        self.entropy_last_epoch = None
        self.loss_last_epoch = None
        
    @staticmethod
    def updatemetric(metric, outputs, labels):
        _ , prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)


    def _get_outputs(self, images):
        
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        
        if self.args.model == 'resnet18':
            return self.model(images)


    def run_epoch(self, cur_epoch, optimizer, n_steps: int) -> None:
        '''
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level.
            -) cur_epoch: current epoch of training;
            -) optimizer: optimizer used for the local training.
        '''
        cumu_loss = 0
        print('Epoch', cur_epoch + 1)
        tot_entropies = np.array([])
        for cur_step, (images, labels) in enumerate(self.train_loader):
                
            # Total steps needed to complete an epoch. Computed as:
            # 
            #           self.n_total_steps = floor(len(self.dataset) / self.args.bs).
            # 
            # Example: len(self.dataset) = 600, self.args.bs = 16, self.n_total_steps = 37
            self.n_total_steps = len(self.train_loader)

            images = images.to(self.device, dtype = torch.float32) 
            labels = labels.to(self.device, dtype = torch.long)
            outputs = self._get_outputs(images) #dim = [bs, 16, h, w]
            
            loss = self.reduction(self.criterion(outputs,labels), labels)
            cumu_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Task 5
            if cur_epoch == self.args.num_epochs-1 and self.args.use_entropy == 'true':
                #Turn logits into probs
                probs = F.softmax(outputs, dim=1) #dim = [bs, 16, h, w]
                batch_entropies = scipy.stats.entropy(probs.cpu().detach().numpy(), axis = 1).reshape(probs.shape[0] ,-1).mean(axis = 1) #dim = [bs,]
                tot_entropies = np.concatenate((tot_entropies, batch_entropies))
                self.entropy_last_epoch = tot_entropies.mean()
                self.loss_last_epoch = cumu_loss /len(self.train_loader)

            if self.args.wandb != None and self.args.framework == 'centralized':
                wandb.log({"batch loss": loss.item()})

            # We keep track of the loss. Notice we are storing the loss for
            # each mini-batch.
            #wandb.log({"loss": loss.mean()})
            
            # We are considering 10 batch at a time. TO DO: define a way to handle different values.
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
                self.n_10th_steps.append(self.count)
                print(f'epoch {cur_epoch + 1} / {self.args.num_epochs}, step {cur_step + 1} / {self.n_total_steps}, loss = {loss.mean():.3f}')
        
        return cumu_loss /len(self.train_loader)
    
    #TODO: funzione da correggere dopo il cambiamento di opt e sched creati on the fly
    def run_epoch_self_train(self, cur_epoch, n_steps):
        cumu_loss = 0
        print('Epoch', cur_epoch + 1)
        for cur_step, (images, _) in enumerate(self.train_loader):
                
            self.n_total_steps = len(self.train_loader)

            #! è corretto che queste immagini abbiano subito le stesse transforms?
            images = images.to(self.device, dtype = torch.float32) 
            outputs = self._get_outputs(images)
            
            loss = self.self_train_loss(outputs, images)


            cumu_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.wandb != None and self.args.framework == 'centralized':
                wandb.log({"batch loss": loss.item()})

            # We keep track of the loss. Notice we are storing the loss for
            # each mini-batch.
            #wandb.log({"loss": loss.mean()})
            
            # We are considering 10 batch at a time. TO DO: define a way to handle different values.
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
                self.n_10th_steps.append(self.count)
                print(f'epoch {cur_epoch + 1} / {self.args.num_epochs}, step {cur_step + 1} / {self.n_total_steps}, loss = {loss.mean():.3f}')
        
        return cumu_loss /len(self.train_loader)

    def _configure_optimizer(self, config):
        """
            Creates an optimizier with the parameters contained in config. Also discard the useless settings.
        """
        opt_config = config.get('optimizer')
        opt_name = opt_config.get('name')
        opt_method = getattr(optim, opt_name)
        opt_signature  = set(signature(opt_method).parameters.keys())
        valid_params_k = opt_signature.intersection(set(opt_config))
        valid_params_dict = {key: opt_config[key] for key in valid_params_k}
        
        optimizer = opt_method(self.model.parameters(), **valid_params_dict)
        if self.args.framework == 'centralized':
            print(optimizer)
        return optimizer
    
    def _configure_scheduler(self, config, optimizer):
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
            scheduler = sch_method(optimizer, **valid_params_dict)
            if self.args.framework == 'centralized':
                print('Scheduler:\n',type(scheduler),"\n", scheduler.state_dict())
        else:
            scheduler = None
            if self.args.framework == 'centralized':
                 print("No scheduler")

        return scheduler
    
    def  configure_opt_sch(self, config: dict):
        """
            Simply call the method to create the optimizer and the scheduler
        """
        #il file config che riceve deve essere un dizionario con chiavi esterne "optimizer" e "scheduler"
        #nel se usi config = wand.config viene automaticamente fatto in questo modo 
        optimizer = self._configure_optimizer(config)
        scheduler = self._configure_scheduler(config, optimizer)
        return optimizer, scheduler

    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def train(self, config, n_steps=10):
        '''
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        '''

        self.model.train()

        optimizer, scheduler = self.configure_opt_sch(config) #Create optimizer and scheduler


        # Training loop. We initialize some empty lists because we need to store the information about the statistics computed
        # on the mini-batches.
        self.n_total_steps = len(self.train_loader)
        self.mean_loss = []
        self.mean_std  = []
        self.n_10th_steps = []
        self.n_epoch_steps = [self.n_total_steps]
        self.count = 0

       
        num_train_samples = len(self.train_dataset)

        # We iterate over the epochs.
        for epoch in range(self.args.num_epochs):
            
            # wandb
            if self.args.wandb != None and self.args.framework == 'centralized':
                    wandb.log({"lr": optimizer.param_groups[0]['lr']})
            
            if self.args.self_train == 'true':
                avg_loss = self.run_epoch_self_train(epoch, n_steps)
            else:
                avg_loss = self.run_epoch(epoch, optimizer, n_steps)

            #if there is a scheduler do a step at each epoch
            if scheduler != None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)
                    print("Reduce On Plateau Scheduler, lr:", optimizer.param_groups[0]['lr'])
                else:
                    scheduler.step()
                    print("Altro Scheduler, lr:", optimizer.param_groups[0]['lr'])
                
            # wandb
            if self.args.wandb != None and self.args.framework == 'centralized':
                wandb.log({"loss": avg_loss, "epoch": epoch})
            
            # Here we are simply computing how many steps do we need to complete an epoch.
            self.n_epoch_steps.append(self.n_epoch_steps[0] * (epoch + 1))

        update = self.generate_update()
        
        return num_train_samples, update, avg_loss



    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(self.test_loader)):
                images = images.to(self.device) 
                labels = labels.to(self.device)
                outputs = self._get_outputs(images)
                self.updatemetric(metric, outputs, labels)
   
    
    def checkImageAndPrediction(self, ix, alpha = 0.4):
        """
        This method plot the image and the predicted mask.
        :param int ix: index of the image in self.dataset
        :param float alpha: transparency index
        """
        self.model.eval()
        with torch.no_grad():
            img = self.train_dataset[ix][0].cuda()
            outputLogit = self.model(img.view(1, 3, 512, 928))['out'][0]
            prediction = outputLogit.argmax(0)
            label2color = Label2Color(idda_16_cmap())
            pred_mask = label2color(prediction.cpu()).astype(np.uint8)
            plt.imshow(denormalize(img.cpu()).permute(1,2,0))
            plt.imshow(pred_mask, alpha = alpha)

    # Method used to extract avg style
    def extract_avg_style(self, b):
        self.avg_style, self.win_sizes = self.style_extractor.extract_avg_style(b=b)
        return self.avg_style, self.win_sizes, self.name
    
    #Method used to extract the style of a single image
    def extract_single_img_style(self, ix, b):
        style = self.style_extractor.extract_style_given_img(ix = ix, b = b)
        return style
    
    
    def print_learning(self, step: int, plot_error = False) -> None:
        '''
        Function used to print the learning curves.
        -) step: how many steps we want to wait until plotting the loss;
        -) plot_error: whether we want to plot the std of the loss.
        '''
        
        # We need arrays in order to plot the results.
        self.mean_loss = np.array(self.mean_loss)
        self.mean_std  = np.array(self.mean_std)

        # We plot vertical lines for each epoch. Notice that the epoch may not be a multiple of the argument step, since
        # len(self.dataset) may not be perfectly divisible by self.args.bs (e.g. step = 10, self.n_total_steps = 37).
        lines = [plt.axvline(_x, linewidth = 1, color = 'g', alpha = 0.75,
                            linestyle = '--') for _x in self.n_epoch_steps]
        
        # We plot the loss curve uing plt.errorbar since we may want to include the std as well.
        markers, caps, bars = plt.errorbar(self.n_10th_steps, self.mean_loss, 
                                            yerr = self.mean_std / np.sqrt(self.args.bs),   # We consider the SE of the mean.
                                            ecolor = 'red', elinewidth = 1.5, 
                                            capsize = 1.5, capthick = 1.0, 
                                            color = 'blue',
                                            marker = 'o', fillstyle = 'none')
        plt.title('Loss')

        # This was done only to plot extra text in the legend, namely the optimizer and the rate decay method we chose.
        extra = Rectangle((0, 0), 1, 1, fc = "w", fill = False, 
                            edgecolor = 'none', linewidth = 0)
        text = f'Optimizer: {self.opt}\nRate decay: {self.sch}'
        plt.legend([extra, markers, lines[0]], (text, 'Loss ± SE', 'Epoch'))

        # We plot a line for each step (e.g. one line each n_steps mini-batches).
        xticks = np.arange(step, self.args.num_epochs * np.floor(len(self.train_dataset) / self.args.bs) , step)
        plt.xticks(xticks)
        plt.grid(axis = 'x', alpha = 0.5)
        
        # We do this in order to consider whether to plot the errors as well.
        [bar.set_alpha(0.3 * plot_error) for bar in bars]
        [cap.set_alpha(0.3 * plot_error) for cap in caps]

        plt.show()

    def set_teacherModel(self, teacherModel, dataloader = True):
        self.teacherModel = teacherModel
        if dataloader:
            self.train_dataset.set_teacherModel(teacherModel)

        print("Teacher model acquired")
    
    def get_entropy_dict(self):
        return {'loss' : self.loss_last_epoch,
                'cluster': self.cluster_id,
                'entropy': self.entropy_last_epoch}