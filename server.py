import copy
from collections import OrderedDict
from centralized import Centralized
import numpy as np
import torch
import wandb


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients #lista con un solo elemento (istanza della classe centralized (test_client = False))
        self.test_clients = test_clients #lista con due elementi (istanza della classe centralized (test_client = True))
        #self.selected_clients = [] #client che vengono selezionati ad ogni round
        self.model = model #da passare poi al client
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        #per setting centralized
        self.params = None
        self.mious = {'eval_train':[], "test_same_dom":[], "test_diff_dom":[]}
        

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        """
        Fa il train sul trainClient (train.txt)
        """
        updates = []
        for i, client in enumerate(self.select_clients()):
            print(client.name)
            self.load_server_model_on_client(client)
            num_samples, update = client.train()

            #client_update = copy.deepcopy(client.model.state_dict())
            updates.append((num_samples, update))

        return updates #una lista di model.state_dict() dei diversi clients
    
    def _aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        if self.args.dataset == 'iddaCB':
            return updates[0][1]
        
        elif self.args.dataset == 'idda':
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
        self.model_params_dict = self._aggregate(updates)


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        """
        Train chiama train_round(), salva il modello restituito da train_round(),
        chiama eval_train (train.txt) che fa l'evaluation sul train client e test()
        che fa l'evaluation su entrambi i test clients (test_same_dom e test_diff_dom)
        """
        for round in range(self.args.num_rounds):
            print(f'round {round+1}')
            #funzione per scegliere m train_clients

            updates = self.train_round() #crea un lista [(num_samples, model_state_dict),...,]           
            
            new_model_parmas = self._aggregate(updates)
            self.model.load_state_dict(new_model_parmas)
            self.model_params_dict = copy.deepcopy(self.model.state_dict())

            #self.update_model(updates) #aggiorna self.model_params_dict
            #self.model.load_state_dict = self.model_params_dict
            #self.model_params_dict = new_state_dict

            self.eval_train()
            self.test()

        


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
        for client in self.train_clients: #tutti i client o solo i selected?
            #print(f"Testing client {client.name}...")
            self.load_server_model_on_client(client)
            client.test(metric)

        miou = metric.get_results()['Mean IoU']
        if self.args.wandb != None:
            wandb.log({'train_miou':miou})

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


    