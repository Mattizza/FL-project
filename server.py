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
from client_selector import ClientSelector

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
            if self.args.self_train == 'true' or self.args.our_self_train: #load a model trained on source dataset
                self.load_source_trained_model()
            else:
                self.load_model_to_test_results() #load a model to test the performances
        
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        if args.dataset != 'gta5':
            self.mious = {'eval_train':[], "test_same_dom":[], "test_diff_dom":[]}
        elif args.dataset == 'gta5':
            self.mious = {'eval_train': [], 'idda_test':[], "test_same_dom":[], "test_diff_dom":[]}
        
        self.client_selector_custom = ClientSelector(self.train_clients)

        #Task 4
        #
        if self.args.self_train == 'true':
            if self.args.checkpoint_to_load == None:
                sys.exit('An error occurred: you must specify a checkpoint to load if in self_train mode!')
            self.teacher_model = model #creo un teacher model allenato
        
        if self.args.our_self_train == 'true':
            if self.args.checkpoint_to_load == None:
                sys.exit('An error occurred: you must specify a checkpoint to load if in self_train mode!')
            self.teacher_model = model #creo un teacher model allenato
        
        #Task 5
        #
        """if self.args.framework == 'federated':
            self.aggregator = Aggregator(self.args.sigma)"""
        
        self.isFirstRound = True
        self.clients_entropy = {}
        self.clients_num_samples = {}

            
    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        
        if self.args.custom_client_selection == 'true':
            print("\nIn custom client selection") #Debugging puropses
            if self.isFirstRound:
                self.isFirstRound = False
                p = None #uniform
                print("\nprobs",p)
                print()
            else:
                
                #Questi due dizionari contengono solo i valori per i client selezionati il round prima oppure di tutti i clients? Sì
                #Questi dizionari dovrebbe resettarsi alla fine di ogni round prima di prendere i nuovi valori? Sì

                """clients_entropy[client.name] = {'loss': avg_loss,
                                                'cluster': client.cluster_id,
                                                'entropy': client.entropy_last_epoch,
                                                }
                clients_num_samples[client.name] = num_samples"""

                print(f"\n\nentropies: {self.clients_entropy}\n")

                p = self.client_selector_custom.compute_probs(self.clients_entropy, self.clients_num_samples)
                
                #resetti i dizionari ad ogni round
                print('resetting dict')#debug
                self.clients_entropy = {}
                self.clients_num_samples = {} 
                for i, c in enumerate(self.train_clients):
                    print(f"{c.name}: {p[i]:.4f}")
                print()
        else: #uniform case
            p = None

        choosen_clients = np.random.choice(self.train_clients, num_clients, replace=False, p = p)

        return choosen_clients
    
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
            
            elif self.args.our_self_train == 'true':
                client.set_teacherModel(self.teacher_model)
            
            num_samples, update , avg_loss= client.train(self.config)

            if self.args.use_entropy == 'true':
                #print(client.name)
                #print("entropies",client.get_entropy_dict())
                #print("len",len(client.train_dataset))
                self.clients_entropy[client.name] = client.get_entropy_dict()
                self.clients_num_samples[client.name] = len(client.train_dataset)

            #client_update = copy.deepcopy(client.model.state_dict())
            updates.append((num_samples, update))
            
            num_samples_each_client.append(num_samples)
            avg_losses.append(avg_loss)
        
        round_avg_loss = np.average(avg_losses, weights = num_samples_each_client)

        #Wandb
        if self.args.wandb != None and self.args.framework == 'federated':
                wandb.log({"round loss": round_avg_loss})

        return updates, round_avg_loss #update is a list of model.state_dict(), each one of a different client

    def train_round_entropy(self):
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

        clients_entropy = {}
        client_num_samples = {}
        client_model = {}
        for i, client in enumerate(self.select_clients()):
            print(client.name)
            self.load_server_model_on_client(client)
            
            if self.args.self_train == 'true':
                client.self_train_loss.set_teacher(self.teacher_model)#passa in teacher model alla loss del client
            
            elif self.args.our_self_train == 'true':
                client.set_teacherModel(self.teacher_model)
            
            num_samples, update , avg_loss= client.train(self.config)

            #!
            #self.clusters = {'T_0_A' : 0, 'T_1_E': 2}
            
            clients_entropy[client.name] = {'loss': avg_loss,
                                            'cluster': client.cluster_id,
                                            'entropy': client.entropy_last_epoch
                                            }
            client_num_samples[client.name] = num_samples

            client_model[client.name] = update

            clusters = np.unique(list(self.clusters.keys()))


            num_samples_each_client.append(num_samples)
            avg_losses.append(avg_loss)

        
        self._newAggregate(clients_entropy, client_num_samples, clusters, self.args.sigma, client_model)
        round_avg_loss = np.average(avg_losses, weights = num_samples_each_client)

        #Wandb
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
        
    
    def _newAggregate(self, clients_entropy, client_num_samples, clusters, sigma, client_model):

        if self.args.framework == 'federated':
            averaged_sol_n = self.aggregator.aggregate(clients_entropy, client_num_samples, clusters, sigma, client_model)

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

        #check_list = [1,5,10] 
        #i = 0
        for round in range(self.args.num_rounds):
            print(f'\nRound {round+1}\n')

            #Task 4
            if self.args.self_train == 'true' and self.args.T != None:
                if round % self.args.T == 0: #ogni T round (e a round 0)
                    self.teacher_model.load_state_dict(self.model_params_dict) #aggiorno il teacher model (lo rendo uguale a global model)

            updates, round_avg_loss = self.train_round() #crea un lista [(num_samples, model_state_dict),...,]           
            
            self.update_model(updates)
        
        """#Check performances every
            if round+1 == check_list[i]:
                if i < len(check_list)-1:
                    i+=1
                print("\nTesting model\n")
                print("Model del server prima del test:", self.model.training)
                print("Model di un train client random prima del test:", self.train_clients[8].model.training)
                print("Model di un test client random prima del test:", self.test_clients[1].model.training)
                
                self.eval_train()
                self.test()
                print("Model del server dopo il test:", self.model.training)
                print("Model di un train client random dopo il test:", self.train_clients[8].model.training)
                print("Model di un test client random dopo il test:", self.test_clients[1].model.training)

                self.model.train()

                print("Model del server dopo mode train:", self.model.training)
                print("Model di un train client random dopo mode train:", self.train_clients[8].model.training)
                print("Model di un test client random dopo mode train:", self.test_clients[1].model.training)"""


                
                                
        
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
            if os.path.exists(path):
                checkpoint = torch.load(path)
                checkpoint['metrics_dict']['eval_train'] = metric
                #checkpoint['target_eval_miou'] = miou
                torch.save(checkpoint, path)
                print("\nAdded eval_train metric in checkpoint\n")

        print(f'Mean IoU: {miou:.2%}')
        self.mious['eval_train'].append(miou)
  

    def test(self):
        """
        This method handles the test on the test clients.
        Load the server model on each test client, reset the previously computed
        metrics, test the model on the test client's dataset
        """
        #Se il checkpoint è abilitato ed esiste lo carico
        checkpoint = None #inizializzo il checkpoint come non esistente
        if self.args.name_checkpoint_to_save != None:
            root1 = 'checkpoints'
            root2 = 'idda'
            path = os.path.join(root1, root2, self.args.name_checkpoint_to_save)
            if os.path.exists(path): #Se il checkpoint esiste lo carico
                    checkpoint = torch.load(path)

        for client in self.test_clients:
            print(f"Testing client {client.name}...")
            self.load_server_model_on_client(client)
            metric = self.metrics[client.name]
            metric.reset()
            client.test(metric)
            miou = metric.get_results()['Mean IoU']

            if checkpoint != None:
                checkpoint['metrics_dict'][client.name] = metric
                torch.save(checkpoint, path)
                print(f"\nAdded {client.name} metric in checkpoint\n")

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
                    "metrics_dict": {}, #these dicts will be filled with the metrics objects in the test() and eval_train() methods
                    #"target_eval_miou": None, #TODO: qui salvare gli oggetti metrics invece di un singolo numero
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
        print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:")
        if self.args.framework == 'centralized':
                  print(f"\n - epochs executed: {checkpoint['actual_epochs_executed']}")

                  
        elif self.args.framework == 'federated':
                  print(f"\n - local epochs executed for each client for each round: {checkpoint['actual_epochs_executed']}"
                        f"\n - rounds executed: {checkpoint['actual_rounds_executed']}")

        print(f"\n - Eval_miou on {checkpoint['eval_dataset']}: {checkpoint['metrics_dict']['eval_train'].get_results()['Mean IoU']:.2%}"
            f"\n - Test_same_Dom miou on {checkpoint['eval_dataset']}: {checkpoint['metrics_dict']['test_same_dom'].get_results()['Mean IoU']:.2%}"
            f"\n - Test_Diff_Dom miou on {checkpoint['eval_dataset']}: {checkpoint['metrics_dict']['test_diff_dom'].get_results()['Mean IoU']:.2%}\n")
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

        if self.args.fda == 'true':
            self.styles_mapping = checkpoint["styles_mapping"]
            self.winSize = checkpoint["winSize"]
    
    
    