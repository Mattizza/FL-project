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
import os
from tqdm import tqdm

from PIL import Image, ImageDraw
import random

from utils.style_applier import StyleApplier


class ServerGTA:
    def __init__(self, args, source_dataset, source_dataset_test, target_clients, test_clients, model, metrics):
        self.args = args
        self.model = model

        self.source_dataset = source_dataset #train on this gta dataset
        self.source_dataset_test = source_dataset_test #test on this gta dataset
        self.target_clients = target_clients #list of target_clients (those who have a idda patition)
        self.test_clients = test_clients #list [idda_test, idda_same_dom, idda_diff_dom]
        self.test_client_gta = Client(self.args, train_dataset=None, test_dataset=self.source_dataset_test, model=self.model, test_client=True)

        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict()) 
        
        self.train_loader = DataLoader(self.source_dataset, batch_size=self.args.bs, shuffle=True, drop_last=True)
        self.test_loader_gta = DataLoader(self.source_dataset_test, batch_size=1, shuffle=False)
        self.mious = {'idda_test':[], 'test_gta5':[],"test_same_dom":[], "test_diff_dom":[]}

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = None
        self.scheduler = None

        self.t = self.args.t #how many epochs between the evaluations
        self.max_eval_miou = 0.0
        self.pretrain_actual_epochs = 0

        #Task 3.4
        self.b = self.args.b #size of the window to apply FDA
        self.style_applier = StyleApplier()
        
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


    def  create_opt_sch(self, config):
        """
            Simply call the method to create the optimizer and the scheduler
        """
        #The config file must be a dict with external keys "optimizer" and "scheduler"

        self.set_optimizer(config)
        self.set_scheduler(config)    



    def run_epoch(self, cur_epoch, n_steps: int) -> None:
        '''
        This method is use to run a single epoch in the train on the source dataset (GTA)
        '''
        cumu_loss = 0
        print('Epoch', cur_epoch)
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
            if self.args.wandb != None:
                wandb.log({"batch loss": loss.item()})

            # === Printing === 
            # We keep track of the loss. Notice we are storing the loss for
            # each mini-batch.
            #wandb.log({"loss": loss.mean()})
            
            # We are considering 10 batch at a time.
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
                print(f'epoch {cur_epoch} / {self.args.num_epochs + self.pretrain_actual_epochs }, step {cur_step + 1} / {self.n_total_steps}, loss = {loss.mean():.3f}')
        
        return cumu_loss / len(self.train_loader)

    def train(self, n_steps=10):
        '''
        This method trains the model on the source dataset (GTA).
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

            actual_epochs_executed =  self.pretrain_actual_epochs + epoch + 1 #the first addend is != 0 only if a checkpoint is loaded
            
            avg_loss = self.run_epoch(actual_epochs_executed, n_steps)

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
            

            #Check the miou on idda_train, same_dom, diff_dom every t epochs
            if actual_epochs_executed % self.t == 0 or actual_epochs_executed == (self.args.num_epochs):
                eval_miou = self.eval_train() #eval on idda_train
                self.test() #test on the test_clients (idda_same_dom, idda_diff_dom)
                self.model.train() #set the model back to train mode

                # ==== Saving the model ====
                #compare current eval_miou with the max_eval_miou
                if eval_miou > self.max_eval_miou and self.args.name_checkpoint_to_save != None:
                    self.test_gta() #test on gta
                    self.model.train() #set the model back to train mode
                    self.max_eval_miou = eval_miou #update max_eval_miou
                    self.save_checkpoint(eval_miou, actual_epochs_executed)



            # Here we are computing how many steps do we need to complete an epoch.
            self.n_epoch_steps.append(self.n_epoch_steps[0] * (epoch + 1))
        
        avg_loss_last_epoch = avg_loss
        return avg_loss_last_epoch
    
    def save_checkpoint(self, eval_miou, actual_epochs_executed):
        
        

        checkpoint = {"model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "target_eval_miou": eval_miou, #miou su idda_train
                    "actual_epochs_executed": actual_epochs_executed,
                    "eval_dataset" : type(self.test_clients[0].test_dataset).__name__,
                    "mious": self.mious, #all the calculated mious, for eval, diff_dom and same_dom
                    "metrics": self.metrics, #these are the updated metrics
                    "train_dataset": type(self.source_dataset).__name__}
        
        checkpoint['scheduler_state'] = self.scheduler.state_dict() if self.scheduler != None else None  #manage the case when the scheduler is none

        if self.args.fda == 'true':
             checkpoint["styles_mapping"] = self.get_styles_mapping()
             checkpoint["winSize"] = self.get_window_sizes()

        root1 = 'checkpoints'
        root2 = 'gta'
        customPath = self.args.name_checkpoint_to_save
        path = os.path.join(root1, root2, customPath)
        torch.save(checkpoint, path)
        print(f"=> Saving checkpoint at {path}.\n"
            f"- Target_eval_miou: {eval_miou:.2%}\n"
            f"- Test_same_dom: {self.mious['test_same_dom'][-1]:.2%}\n"
            f"- Test_diff_dom: {self.mious['test_diff_dom'][-1]:.2%}\n"
            f"- Test_gta: {self.mious['test_gta5'][-1]:.2%}\n")
            
    
    def load_checkpoint(self):
        root1 = 'checkpoints'
        root2 = 'gta'
        path = os.path.join(root1, root2, self.args.checkpoint_to_load)
        checkpoint = torch.load(path)
        print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:\n"
            f" - epochs executed: {checkpoint['actual_epochs_executed']}\n"
            f" - Target_eval_miou on {checkpoint['eval_dataset']}: {checkpoint['target_eval_miou']:.2%}\n"
            f" - Test_same_dom: {checkpoint['mious']['test_same_dom'][-1]:.2%}\n"
            f" - Test_diff_dom: {checkpoint['mious']['test_diff_dom'][-1]:.2%}\n"
            f" - Test_gta: {checkpoint['mious']['test_gta5'][-1]:.2%}\n")
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state']) if checkpoint['scheduler_state'] != None else None
        
        self.max_eval_miou = checkpoint['target_eval_miou']
        self.pretrain_actual_epochs = checkpoint['actual_epochs_executed']
        self.mious = checkpoint['mious']

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
        print(f"\nEvaluating on the train set of Idda")
        client = self.test_clients[0]
        self.load_server_model_on_client(client)
        client.test(metric)

        miou = metric.get_results()['Mean IoU']
        
        # wandb
        if self.args.wandb != None:
            wandb.log({'idda_train_miou':miou})

        print(f'Mean IoU on idda_eval: {miou:.2%} \n')
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
            print(f'Mean IoU: {miou:.2%} \n')
            self.mious[client.name].append(miou)
    
    def test_gta(self):
        metric = self.metrics['test_gta5']
        metric.reset()
        print(f"Evaluating on gta train dataset")
        client = self.test_client_gta
        self.load_server_model_on_client(client)
        client.test(metric)
        miou = metric.get_results()['Mean IoU']
        # wandb
        if self.args.wandb != None:
            wandb.log({'gta_miou': miou})
        print(f'Mean IoU on gta_train: {miou:.2%} \n')
        self.mious['test_gta5'].append(miou)
        return miou



    def load_server_model_on_client(self, client):
        client.model = self.model #<- con questa passi proprio il modello
    
    """def apply_styles(self):
        self.source_dataset.set_style_tf_fn(self.styleaug.apply_style)"""

    def compare_wo_w_style(self, ix = None):
        #ix = 288
        if ix == None:
            ix = random.randint(0, 400)
        print(ix)

        # Open the two images
        self.source_dataset.return_original = True
        image1 = self.source_dataset[ix][0]
        self.source_dataset.return_original = False
        self.source_dataset.apply_only_fda = True
        image2 = self.source_dataset[ix][0]

        # Resize the images to have the same height
        height = max(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * height / image1.height), height))
        image2 = image2.resize((int(image2.width * height / image2.height), height))

        # Calculate the width for the white column
        column_width = 10

        # Create a new image with adjusted width
        result_width = image1.width + column_width + image2.width
        result = Image.new('RGB', (result_width, height))

        # Paste the images with a white column
        result.paste(image1, (0, 0))
        result.paste((255, 255, 255), (image1.width, 0, image1.width + column_width, height))
        result.paste(image2, (image1.width + column_width, 0))

        self.source_dataset.apply_only_fda = False

        # Display the result image
        result.show()

        return result #result is a pil image
    
    """def delete_styles(self):
        self.styleaug.delete_styles()"""

    """def list_styles(self):
        print("These are the styles available:")
        print(self.styleaug.styles_names)"""

    def load_styles(self):
        #server make the clients extract the avg_style and pass them to him
        #styles are loaded in the style_applier's bank
        
        for target_client in self.target_clients:
            client_style, win_sizes, styles_name = target_client.extract_avg_style(b = self.b) 
            self.style_applier.add_style_to_bank(client_style, styles_name)
            if self.style_applier.sizes == None:
                self.style_applier.set_win_sizes(win_sizes)

    def apply_styles(self): #here we pass a funcion to the dataset that will be used to apply the styles (as a transform)
        self.source_dataset.set_style_tf_fn(self.style_applier.apply_style)

    def get_styles_mapping(self):
        return {"style": self.style_applier.styles_bank, "id": self.style_applier.styles_names}
    
    def get_window_sizes(self):
        return self.style_applier.sizes
    