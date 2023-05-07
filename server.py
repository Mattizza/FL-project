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
        self.model = model #da passare poi al client
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        #per setting centralized
        self.params = None
        self.mious = {'eval_train':[], "test_same_dom":[], "test_diff_dom":[]}

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients, config):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        """
        Fa il train sul trainClient (train.txt)
        """
        updates = []
        for i, client in enumerate(clients):
            self.load_server_model_on_client(client)
            client.train(config)
            client_update = copy.deepcopy(client.model.state_dict())
            updates.append(client_update)
        return updates #una lista di model.state_dict() dei diversi clients

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        if self.args.dataset == 'iddaCB':
            return updates[0]
        
        elif self.args.dataset == 'idda':
            raise NotImplementedError

    def train(self, config):
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
            updates = self.train_round(self.train_clients, config)
            new_state_dict = self.aggregate(updates)
            self.model_params_dict = new_state_dict

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
        for client in self.train_clients:
            print(f"Testing client {client.name}...")
            self.load_server_model_on_client(client)
            client.test(metric)

        miou = metric.get_results()['Mean IoU']
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
            wandb.log({client.name : miou})
            print(f'Mean IoU: {miou:.2%}')
            self.mious[client.name].append(miou)


    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)


    