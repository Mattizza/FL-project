import numpy as np

class AggWeightCalculator():


    def __init__(self, args):
 
        self.args = args
        self.tot_num_cluster = None
        self.alpha = self.args.alpha_weight_agg     #num_samples weight
        self.beta = self.args.beta_weight_agg       #entropy weight
        self.gamma = 1 - self.alpha - self.beta     #Complement to 1. loss weight
        self.llambda = self.args.llambda_weight_agg #lambda parameter in the sigmoid function (adjust the slope of the sigmoid)
    
    def set_tot_num_cluster(self, tot_num_cluster):
        self.tot_num_cluster = tot_num_cluster

    def get_weight(self, selected_clients, metric = 'loss'):
        """
        get the weight for each client based on the metric (could be loss or entropy)
        """
        
        #Each client has as attribute cluster_id, entropy_last_epoch and loss_last_epoch
        
        if self.tot_num_cluster == None:
            raise Exception('tot_num_cluster must be set before calling get_weight')

        metrics = [] #[loss1, loss2, loss3, loss4, loss5]

        clusters_of_selected_clients = [] # will be [0, 1, 0, 4, 1]

        metrics_by_cluster = [[] for _ in range(self.tot_num_cluster)] # will be [[loss1, loss3], [loss2, loss5], [], [], [loss4]}, the index is the cluster id

        for client in selected_clients: # iterate on all the selected clients (Client object)
            
            clusters_of_selected_clients.append(client.cluster_id) #save the client's cluster id
            
            if metric == 'loss':
                #debugging
                print(f'loss {client.name}: {client.get_entropy_dict()}')
                metrics.append(client.get_entropy_dict()['loss']) #save the client's loss of the last epoch
            elif metric == 'entropy':
                metrics.append(client.get_entropy_dict()['entropy']) #save the client's entropy of the last epoch
        
        unique_clusters = set(clusters_of_selected_clients)
        if len(unique_clusters) == 1: #If there is a single cluster, all the clients have the same weight
            return [1/len(selected_clients)]*len(selected_clients)

        #debugg
        if metric == 'loss':
            print('\nloss: ', metrics)
        elif metric == 'entropy':
            print('\nentropy: ', metrics)

        num_clients_in_cluster = [0]*self.tot_num_cluster
        for client_ix, cluster_id in enumerate (clusters_of_selected_clients):
            num_clients_in_cluster[cluster_id] += 1
            metrics_by_cluster[cluster_id].append(metrics[client_ix])
        
        #Calculate the mean of the losses for each cluster
        cluster_metrics = []
        for cluster_id in range(self.tot_num_cluster):
            if len(metrics_by_cluster[cluster_id]) != 0:
                cluster_metrics.append(sum(metrics_by_cluster[cluster_id])/len(metrics_by_cluster[cluster_id]))
            else:
                cluster_metrics.append(np.nan)
        
        #Calcola the mean of the clusters'mean
        mean_cluster_metric = np.nanmean(cluster_metrics)

        std_cluster_metric = np.nanstd(cluster_metrics)

        sigmoid = lambda x: 1/(1+np.exp(-self.llambda * x))
        cluster_metrics_sigmoid = sigmoid((cluster_metrics - mean_cluster_metric)/std_cluster_metric)

        tot_cluster_metrics_sigmoid = np.nansum(cluster_metrics_sigmoid)

        clusters_w = cluster_metrics_sigmoid/tot_cluster_metrics_sigmoid
        clients_w = [clusters_w[cluster_of_client] / num_clients_in_cluster[cluster_of_client] for cluster_of_client in clusters_of_selected_clients ]
        return clients_w
    
    def get_weight_num_samples(self, selected_clients):
        """
        get the weight for each client based on the number of samples
        """
        num_samples_per_client = np.array([client.num_train_samples for client in selected_clients])
        tot_num_samples = np.sum(num_samples_per_client)
        return num_samples_per_client/tot_num_samples
        
    def get_final_w(self, selected_clients):
        """
        get the final weight for each client
        """
        clients_w_loss = 0
        clients_w_entropy = 0
        clients_w_num_samples = 0
        
        if self.alpha != 0:
            clients_w_num_samples =  np.array(self.get_weight_num_samples(selected_clients))

        if self.beta != 0:
            clients_w_entropy =  np.array(self.get_weight(selected_clients, metric = 'entropy'))
            print('clients_w_entropy: ', clients_w_entropy)
        
        if self.gamma != 0:
            clients_w_loss = np.array(self.get_weight(selected_clients, metric = 'loss'))
            print('clients_w_loss: ', clients_w_loss)

        clients_final_w = self.alpha * clients_w_num_samples + self.beta * clients_w_entropy + self.gamma * clients_w_loss 
        print('clients_final_w: ', clients_final_w)
        
        return clients_final_w