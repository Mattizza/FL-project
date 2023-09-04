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
from utils.client_selector import ClientSelector
import pandas as pd

class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics, config):
        self.args = args
        self.train_clients = train_clients #list with on obj (class Client (test_client = False))
        self.test_clients = test_clients #list with two objs (class Client (test_client = True))
        self.model = model #will be passed to clients

        self.prev_rounds_executed = 0 #useful if continue training from a checkpoint
        
        self.config = config

        self.mious = {'round':[], 'eval_train':[], "test_same_dom":[], "test_diff_dom":[]}
        
        # ==== Loading the checkpoint if enabled====
        if self.args.checkpoint_to_load != None:
            print("Continue training from a checkpoint...")
            #task 4
            if self.args.self_train == 'true':
                self.model.eval()
                self.teacher_model = copy.deepcopy(model) #initialize a teacher model in eval mode (it is not pretrained yet)
                self.model.train()

            self.load_model_to_continue_train_task_2_4() #load a model to continue the training

        #Task 4: if a checkpoint to continue the training is not defined, a source trained model is loaded from a checkpoint
        if self.args.self_train == 'true':
            if self.args.source_trained_ckpt != None and self.args.checkpoint_to_load == None:
                print("Loading a source trained model...")
                self.load_source_trained_model()
                self.model.eval()
                self.teacher_model = copy.deepcopy(model) #create a pre-trained teacher model
                self.model.train()
            
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.client_selector_custom = ClientSelector(self.train_clients, beta = self.args.beta, llambda = self.args.llambda)

        if self.args.our_self_train == 'true': #pseudo labels before transforms
            if self.args.checkpoint_to_load == None:
                sys.exit('An error occurred: you must specify a checkpoint to load if in self_train mode!')
            self.teacher_model = copy.deepcopy(model) #create a trained teacher model
        
        self.num_teacher_updates = 0

        #Task 5
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

                """clients_entropy[client.name] = {'loss': avg_loss,
                                                'cluster': client.cluster_id,
                                                'entropy': client.entropy_last_epoch,
                                                }
                clients_num_samples[client.name] = num_samples"""

                print(f"\n\nentropies: {self.clients_entropy}\n")

                p = self.client_selector_custom.compute_probs(self.clients_entropy, self.clients_num_samples)
                
                #reset dict at every round
                print('resetting dict')#debugging purposes
                self.clients_entropy = {}
                self.clients_num_samples = {} 
                for i, c in enumerate(self.train_clients):
                    print(f"{c.name}: {p[i]:.4f}")
                print()
        else: #uniform case
            p = None

        choosen_clients = np.random.choice(self.train_clients, num_clients, replace=False, p = p)

        return choosen_clients


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

            #Task4 - At each round, the teacher model is setted in the selected clients
            if self.args.self_train == 'true':
                client.self_train_loss.set_teacher(self.teacher_model)#pass the teacher model to the client loss
            
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
                m_tot_samples += client_samples #tot num of data point in the selected clients

                for key, value in client_model.items():
                    if key in base:
                        base[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base[key] = client_samples * value.type(torch.FloatTensor)
            averaged_sol_n = copy.deepcopy(self.model_params_dict)
            for key, value in base.items():
                if m_tot_samples != 0:
                    averaged_sol_n[key] = value.to('cuda') / m_tot_samples
            
            return averaged_sol_n #is a model_state_dict
        

    def update_model(self, updates):
        #calls aggregate() that returns a new_state_dict
        new_model_parmas = self._aggregate(updates)
        self.model.load_state_dict(new_model_parmas)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        
        call train_round(), save the model returned by train_round(),
        call eval_train (train.txt) that execute the evaluation on train client and test()
        that test on both test clients (test_same_dom e test_diff_dom)
        """

        # wandb in federated
        if self.args.framework == 'federated' and self.args.wandb != None:
            wandb.log({"Num. clients per round": self.args.clients_per_round}, commit = False)
            wandb.log({"Num. local epochs": self.args.num_epochs}, commit = False)
            wandb.log({"Num. rounds": self.args.num_rounds}, commit = False)

        #All the following code of this function is for decide when to test and update the teacher model if in task 4
        self.check_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100] 
        i = 0
        for round in range(self.args.num_rounds): 
            true_round = round + 1 + self.prev_rounds_executed #self.prev_rounds_executed is 0 if you don't load a checkpoint
            print(f'\nRound {true_round}\n')

            #Task 4 - updating the teacher model
            if self.args.self_train == 'true' and self.args.T != None:
                if round % self.args.T == 0: #ogni T round (e a round 0)
                    print(f"Begin of round {true_round}. Updating teacher model...")
                    self.teacher_model.load_state_dict(self.model_params_dict) #I update the teacher model (making it equal to the global model)(the mode of the teacher model stays eval)
                    self.num_teacher_updates += 1

            updates, round_avg_loss = self.train_round() #updates = [(num_samples, model_state_dict),...,]           
            
            self.update_model(updates)
            
            #Check performances and save checkpoint every r rounds
            if self.args.r == -1: #if you want to use a list of rounds
            
                if round+1 == self.check_list[i]:
                    # Save checkpoint if enabled (for task 2 and 4)
                    if self.args.framework == 'federated' and self.args.name_checkpoint_to_save != None:
                        self.save_checkpoint(self.args.num_epochs, true_round)

                    print("\nTesting...\n")
                    self.eval_train() #evaluation on train clients
                    self.test() #testing on same dom and diff dom
                    self.model.train()

                    if i < len(self.check_list)-1:
                        i+=1
            
            elif self.args.r == None:
                
                if round+1 == self.args.num_rounds: #if we are in the last round and r is not specified
                    # Save checkpoint if enabled (for task 2 and 4)
                    if self.args.framework == 'federated' and self.args.name_checkpoint_to_save != None:
                        self.save_checkpoint(self.args.num_epochs, true_round) # save checkpoint and test only at last epoch
                    
                    print("\nTesting...\n")
                    self.eval_train() #evaluation on train clients
                    self.test() #testing on same dom and diff dom
                    self.model.train()

            else: #when r is != -1 and != None
                
                if true_round % self.args.r == 0:
                    # Save checkpoint if enabled (for task 2 and 4)
                    if self.args.framework == 'federated' and self.args.name_checkpoint_to_save != None:
                        self.save_checkpoint(self.args.num_epochs, true_round)

                    print("\nTesting...\n")
                    self.eval_train() #evaluation on train clients
                    self.test() #testing on same dom and diff dom
                    self.model.train()
            
        #==== After the trainig save the checkpoint if not alredy done ====
        #If we are in the last round, it is not a multiple of r, r is not None, and it is not the last value in the check_list."
        if self.args.r != None and self.args != -1: #has to be done before to handle none case
            if (self.args.num_rounds + self.prev_rounds_executed) % self.args.r != 0:
                # Save checkpoint if enabled (for task 2 and 4)
                if self.args.framework == 'federated' and self.args.name_checkpoint_to_save != None:
                    self.save_checkpoint(self.args.num_epochs, self.args.num_rounds + self.prev_rounds_executed)
                
                print("\nTesting...\n")
                self.eval_train() #evaluation on train clients
                self.test() #testing on same dom and diff dom
                self.model.train()

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
        print(f"\nTesting train clients")
        for client in self.train_clients:
            self.load_server_model_on_client(client)
            client.test(metric)

        miou = metric.get_results()['Mean IoU']
        self.mious['eval_train'].append(miou)

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
                checkpoint['mious_dict'] = self.mious
                torch.save(checkpoint, path)
                print("Added eval_train metric in checkpoint")

        print(f'Mean IoU: {miou:.2%}')
        
  

    def test(self):
        """
        This method handles the test on the test clients.
        Load the server model on each test client, reset the previously computed
        metrics, test the model on the test client's dataset
        """
        #If the checkpoint is enabled and it exists, load it.
        checkpoint = None #initialize the checkpoint as non-existent
        if self.args.name_checkpoint_to_save != None:
            root1 = 'checkpoints'
            root2 = 'idda'
            path = os.path.join(root1, root2, self.args.name_checkpoint_to_save)
            if os.path.exists(path): #if the checkpoint exists, I load it
                    checkpoint = torch.load(path)

        for client in self.test_clients:
            print(f"\nTesting client {client.name}...")
            self.load_server_model_on_client(client)
            metric = self.metrics[client.name]
            metric.reset()
            client.test(metric)
            miou = metric.get_results()['Mean IoU']
            self.mious[client.name].append(miou)

            if checkpoint != None:
                checkpoint['metrics_dict'][client.name] = metric
                checkpoint['mious_dict'] = self.mious
                torch.save(checkpoint, path)
                print(f"Added {client.name} metric in checkpoint")

            if self.args.wandb != None:
                wandb.log({client.name : miou})

            print(f'Mean IoU: {miou:.2%}')
            


    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)
    

    def save_checkpoint(self, epochs = None, rounds = None):

        self.mious['round'].append(rounds)

        checkpoint = {"model_state": self.model.state_dict(),
                    "actual_epochs_executed": epochs,
                    "actual_rounds_executed": rounds,
                    "num_clients_per_round": self.args.clients_per_round,
                    "metrics_dict": {}, #these dicts will be filled with the metrics objects in the test() and eval_train() methods
                    "mious_dict": self.mious,
                    "eval_dataset" : type(self.train_clients[0].test_dataset).__name__,
                    "train_dataset": type(self.test_clients[0].test_dataset).__name__,
                    "framework": self.args.framework}
        #Task4
        if self.args.self_train == 'true':
            checkpoint['teacher_model_state'] = self.teacher_model.state_dict()
            checkpoint['num_teacher_updates'] = self.num_teacher_updates
        
        root1 = 'checkpoints'
        root2 = 'idda'
        customPath = self.args.name_checkpoint_to_save
        path = os.path.join(root1, root2, customPath)
        torch.save(checkpoint, path)
        
        print(f"\n=> Saving checkpoint at {path}.\n"
              f" - framework: {checkpoint['framework']}\n"
              f" - rounds executed: {checkpoint['actual_rounds_executed']}\n"
              f" - num clients per round: {checkpoint['num_clients_per_round']}\n"
              f" - epochs executed in each client: {checkpoint['actual_epochs_executed']}"
              )
        if self.args.self_train == 'true':
            print(f" - Num teacher updates: {checkpoint['num_teacher_updates']}")

    def checkpoint_recap(self, checkpoint = None):
        if checkpoint == None:
            root1 = 'checkpoints'
            root2 = 'idda'
            path = os.path.join(root1, root2, self.args.name_checkpoint_to_save)
            checkpoint = torch.load(path)

        print("\nCheckpoint recap:")
        print(f" - framework: {checkpoint['framework']}")

        if self.args.framework == 'centralized':
                  print(f" - epochs executed: {checkpoint['actual_epochs_executed']}")

                  
        elif self.args.framework == 'federated':
                  print(f" - rounds executed: {checkpoint['actual_rounds_executed']}\n"
                        f" - num clients per round: {checkpoint['num_clients_per_round']}\n"
                        f" - local epochs executed for each client for each round: {checkpoint['actual_epochs_executed']}")

        print(f" - Eval_miou: {checkpoint['metrics_dict']['eval_train'].get_results()['Mean IoU']:.2%}"
              f"\n - Test_same_Dom miou: {checkpoint['metrics_dict']['test_same_dom'].get_results()['Mean IoU']:.2%}"
              f"\n - Test_Diff_Dom miou: {checkpoint['metrics_dict']['test_diff_dom'].get_results()['Mean IoU']:.2%}"
              )
        if self.args.self_train == 'true':
            print(f" - Num teacher updates: {checkpoint['num_teacher_updates']}")

    def load_model_to_continue_train_task_2_4(self):
        root1 = 'checkpoints'
        root2 = 'idda'
        path = os.path.join(root1, root2, self.args.checkpoint_to_load)
        checkpoint = torch.load(path)
        print(f"\n=> Loading the following checkpoint:")
        self.checkpoint_recap(checkpoint)
        self.model.load_state_dict(checkpoint['model_state'])
        self.prev_rounds_executed = checkpoint['actual_rounds_executed']
        self.mious = checkpoint['mious_dict']
        if self.args.self_train == 'true':
            self.teacher_model.load_state_dict(checkpoint['teacher_model_state'])
            self.num_teacher_updates = checkpoint['num_teacher_updates']

 
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
        path = os.path.join(root1, root2, self.args.source_trained_ckpt)
        checkpoint = torch.load(path)
        print(f"\n=> Loading the model trained on {checkpoint['train_dataset']}:"
                  f"\n - epochs executed: {checkpoint['actual_epochs_executed']}"
                  f"\n - Target_eval_miou on {checkpoint['eval_dataset']}: {checkpoint['target_eval_miou']:.2%}"
                  f"\n - idda same_dom_miou: {checkpoint['metrics']['test_same_dom'].get_results()['Mean IoU']:.2%}"
                  f"\n - idda diff_dom_miou: {checkpoint['metrics']['test_diff_dom'].get_results()['Mean IoU']:.2%}\n")
        
        self.model.load_state_dict(checkpoint['model_state'])

        if self.args.fda == 'true': #These are needed for the clustering
            self.styles_mapping = checkpoint["styles_mapping"]
            self.winSize = checkpoint["winSize"]

    def download_mious_as_csv(self):
        """
        This method downloads the mious in a csv file
        """
        dict_to_csv = self.mious
        df = pd.DataFrame(dict_to_csv)
        root = 'csv_mious'
        epochs = str(self.args.num_epochs)
        rounds = str(self.args.num_rounds + self.prev_rounds_executed)
        num_clients = str(self.args.clients_per_round)
        task = "4" if self.args.self_train == 'true' else "2"
        file = 'mious_cpr'+num_clients+'e_' + epochs + '_r' + rounds+'_t'+task+'.csv'
        path = os.path.join(root,file)
        df.to_csv(path, index=False)
        print("Saved results in csv file ", path)    