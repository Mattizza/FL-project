import copy
from collections import OrderedDict
from client import Client
import numpy as np
import torch
import wandb
#from utils import definePath
import os
import sys
from tqdm import tqdm

class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics, config):
        self.args = args
        self.train_clients = train_clients #lista con un solo elemento (istanza della classe client (test_client = False))
        self.test_clients = test_clients #lista con due elementi (istanza della classe client (test_client = True))
        #self.selected_clients = [] #client che vengono selezionati ad ogni round
        self.model = model #da passare poi al client

        self.config = config
        
        # ==== Loading the checkpoint if enabled====
        if self.args.checkpoint_to_load != None:
            if self.args.self_train == 'true': #load a model trained on source dataset
                self.load_source_trained_model()
            else:
                self.load_model_to_test_results() #load a model to test the performances
        
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        if args.dataset != 'gta5':
            self.mious = {'eval_train':[], "test_same_dom":[], "test_diff_dom":[]}
        elif args.dataset == 'gta5':
            self.mious = {'eval_train': [], 'idda_test':[], "test_same_dom":[], "test_diff_dom":[]}
        
        

        #Task 4
        #
        if self.args.self_train == 'true':
            if self.args.checkpoint_to_load == None:
                sys.exit('An error occurred: you must specify a checkpoint to load if in self_train mode!')
            self.teacher_model = model #creo un teacher model allenato
            

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)
    
    #! Metodo commentato perchè si crea un optimizer ogni volta che viene chiamato un client
    #def distribute_config_dict(self, config: dict):
        """
            This method iterates over each train client and creates in each of them and optimizer and
            a scheduler according to the configuration contained in config 
        """
        #for c in self.train_clients:
            #c.create_opt_sch(config)


    def train_round(self):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        """
        Fa il train sul trainClient (train.txt)
        """
        num_samples_each_client = []
        updates = []
        avg_losses = []
        for i, client in enumerate(self.select_clients()):
            print(client.name)
            self.load_server_model_on_client(client)
            
            if self.args.self_train == 'true':
                client.self_train_loss.set_teacher(self.teacher_model)#passa in teacher model alla loss del client
            
            num_samples, update , avg_loss= client.train(self.config)

            #client_update = copy.deepcopy(client.model.state_dict())
            updates.append((num_samples, update))
            
            num_samples_each_client.append(num_samples)
            avg_losses.append(avg_loss)
        
        round_avg_loss = np.average(avg_losses, weights = num_samples_each_client)

        if self.args.wandb != None and self.args.framework == 'federated':
                wandb.log({"round loss": round_avg_loss})

        return updates, round_avg_loss #update is a list of model.state_dict(), each one of a different client
    
    def _aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        if self.args.framework == 'centralized':
            return updates[0][1]
        
        elif self.args.framework == 'federated':
            m_tot_samples = 0
            base = OrderedDict()

            for (client_samples, client_model) in updates:
                m_tot_samples += client_samples #numero totale di data points tra i clienti scelti 

                for key, value in client_model.items():
                    if key in base:
                        base[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base[key] = client_samples * value.type(torch.FloatTensor)
            averaged_sol_n = copy.deepcopy(self.model_params_dict)
            for key, value in base.items():
                if m_tot_samples != 0:
                    averaged_sol_n[key] = value.to('cuda') / m_tot_samples
            
            return averaged_sol_n #è un model_state_dict
    
    def update_model(self, updates):
        #chiama aggregate() che restituisce new_state_dict
        new_model_parmas = self._aggregate(updates)
        self.model.load_state_dict(new_model_parmas)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        
        Train chiama train_round(), salva il modello restituito da train_round(),
        chiama eval_train (train.txt) che fa l'evaluation sul train client e test()
        che fa l'evaluation su entrambi i test clients (test_same_dom e test_diff_dom)
        """

        # wandb in federated
        if self.args.framework == 'federated' and self.args.wandb != None:
            wandb.log({"Num. clients per round": self.args.clients_per_round}, commit = False)
            wandb.log({"Num. local epochs": self.args.num_epochs}, commit = False)
            wandb.log({"Num. rounds": self.args.num_rounds}, commit = False)

        #round_min_loss = float('inf')
        
        #Penso sia superfluo perchè il teacher model è già allenato
        """if self.args.self_train == "true" and self.args.T == None:
            self.teacher_model.load_state_dict(self.model_params_dict) #aggiorno il teacher model (lo rendo uguale a global model)
        """
 
        for round in range(self.args.num_rounds):
            print(f'\nRound {round+1}\n')

            #Task 4
            if self.args.self_train == 'true' and self.args.T != None:
                if round % self.args.T == 0: #ogni T round (e a round 0)
                    self.teacher_model.load_state_dict(self.model_params_dict) #aggiorno il teacher model (lo rendo uguale a global model)

            updates, round_avg_loss = self.train_round() #crea un lista [(num_samples, model_state_dict),...,]           
            
            self.update_model(updates)
        
        #==== After the trainig save the checkpoint if enabled ====
        if self.args.name_checkpoint_to_save != None:
            self.save_checkpoint(self.args.num_epochs, self.args.num_rounds)

        print("\nTraining finished!")

    def eval_train(self):
        """
        This method handles the evaluation on the train clients.
        Reset the metrics computed at the previous round, load the model on each
        train client, test the model on the client dataset, update the
        StreamMetric (SM) object. (Note: there is just a single SM obj for all the
        training clients).
        """

        metric = self.metrics['eval_train']
        metric.reset()
        print(f"Testing train clients")
        for client in self.train_clients:
            #print(f"Testing client {client.name}...")
            self.load_server_model_on_client(client)
            client.test(metric)

        miou = metric.get_results()['Mean IoU']
        if self.args.wandb != None:
            wandb.log({'train_miou':miou})
        
        #Set eval_miou on checkpoint if enabled
        if self.args.name_checkpoint_to_save != None:
            root1 = 'checkpoints'
            root2 = 'idda'
            path = os.path.join(root1, root2, self.args.name_checkpoint_to_save)
            checkpoint = torch.load(path)
            checkpoint['target_eval_miou'] = miou
            torch.save(checkpoint, path)
            print("\nAdded eval_miou in checkpoint\n")

        print(f'Mean IoU: {miou:.2%}')
        self.mious['eval_train'].append(miou)
  

    def test(self):
        """
        This method handles the test on the test clients.
        Load the server model on each test client, reset the previously computed
        metrics, test the model on the test client's dataset
        """
        for client in self.test_clients:
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
        client.model.load_state_dict(self.model_params_dict)
    

    def save_checkpoint(self, epochs = None, rounds = None):
        
        checkpoint = {"model_state": self.model.state_dict(),
                    "actual_epochs_executed": epochs,
                    "actual_rounds_executed": rounds,
                    "target_eval_miou": None,
                    "eval_dataset" : type(self.train_clients[0].test_dataset).__name__,
                    "train_dataset": type(self.test_clients[0].test_dataset).__name__}
        
        root1 = 'checkpoints'
        root2 = 'idda'
        customPath = self.args.name_checkpoint_to_save
        path = os.path.join(root1, root2, customPath)
        torch.save(checkpoint, path)
        
        print(f"\n=> Saving checkpoint at {path}.\n")


    def load_model_to_test_results(self):
        root1 = 'checkpoints'
        root2 = 'idda'
        path = os.path.join(root1, root2, self.args.checkpoint_to_load)
        checkpoint = torch.load(path)
        if self.args.framework == 'centralized':
            print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:"
                  f"\n - epochs executed: {checkpoint['actual_epochs_executed']}"
                  f"\n - Target_eval_miou on {checkpoint['eval_dataset']}: {checkpoint['target_eval_miou']:.2%}\n")
        elif self.args.frameworf == 'federated':
            print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:"
                  f"\n - local epochs executed: {checkpoint['actual_epochs_executed']}"
                  f"\n - rounds executed: {checkpoint['actual_rounds_executed']}"
                  f"\n - Target_eval_miou on {checkpoint['eval_dataset']}: {checkpoint['target_eval_miou']:.2%}\n")

        self.model.load_state_dict(checkpoint['model_state'])
    

    def load_source_trained_model(self):
        root1 = 'checkpoints'
        root2 = 'gta'
        path = os.path.join(root1, root2, self.args.checkpoint_to_load)
        checkpoint = torch.load(path)
        print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:"
                  f"\n - epochs executed: {checkpoint['actual_epochs_executed']}"
                  f"\n - Target_eval_miou on {checkpoint['eval_dataset']}: {checkpoint['target_eval_miou']:.2%}\n")
        
        self.model.load_state_dict(checkpoint['model_state'])
    