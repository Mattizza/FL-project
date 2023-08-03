from collections import OrderedDict
import torch
import copy

class Aggregator():


    def __init__(self, aggregation: str = 'FedAvg'):
        '''
        Initialize an Aggregator object.

        Parameters
        ---
        aggregation : str, default 'FedAvg'
            Aggregation technique.
        
        Output
        ---
        Returns an Aggregator object.
        '''

        self.aggregation = aggregation


    def get_weights_entropy(self, client_entropy: dict, clusters: list) -> dict:
        '''
        Compute the weights with respect to the magnitude of the cross entropy over the total.

        Parameters
        ---
        client_entropy : dict
            `dict` object containing the clients as keys and as a values another `dict`
            containing the value of the entropy and the corresponding cluster;
        >>> client_entropy = {'client_1' : {'entropy' : 0.44,
        >>>                                 'cluster' : 'cluster_1'}}
        
        cluster_entropy : dict
            `dict` object containing the clusters as keys and as values the sum of the entropies
            of the corresponding clients;
        >>> {'cluster_1': 0.99, 
        >>>  'cluster_2': 0.09}

        clients : list
            A `list` containing `str` values corresponding to the id of the clients.
        >>> clients = ['client_1', 'client_2']

        clusters : list
            A `list` containing `str` values corresponding to the id of the clusters.
        >>> clusters = ['cluster_1', 'cluster_2']

        Output
        ---
        Returns a `tuple` containing the weights of the clients:  `{clients : weights}` 
        '''

        # Initialize a dictionary {client : weight}.
        
        clientsName = client_entropy.keys()        
        client_weight = {client: 0 for client in clientsName}
        cluster_card = {cluster: 0 for cluster in clusters}

        clusters_entropy = self._compute_clusters_entropy(client_entropy, clusters)

        tot_entropy = sum(list(clusters_entropy.values()))

        # Iterate over the clients and assign the weight corresponding to the relative entropy of the cluster.
        for client in client_entropy.keys():
    
            # Access the client     # Access the entropy of the cluster         
            client_weight[client] = clusters_entropy[client_entropy[client]['cluster']] / tot_entropy
            cluster_card[client_entropy[client]['cluster']] += 1

        for client in client_entropy.keys():

            client_weight[client] = client_weight[client] / cluster_card[client_entropy[client]['cluster']]


        return client_weight

    def get_weights_image(self, client_num_samples):
        tot_img = sum(list(client_num_samples.values()))
        client_img_weight = {c: 0 for c in client_num_samples.keys()}

        for c in client_num_samples.keys():
            client_img_weight[c] = client_num_samples[c] / tot_img

        return client_img_weight

    def _compute_clusters_entropy(self, client_entropy: dict, clusters: list) -> dict:
        '''
        Computes the total entropy of each cluster.

        Parameters
        ---
        client_entropy : dict
            `dict` object containing the clients as keys and as a values another `dict`
            containing the value of the entropy and the corresponding cluster;
        >>> client_entropy = {'client_1' : {'entropy' : 0.44,
        >>>                   'cluster' : 'cluster_1'}}

        clusters : list
            A `list` containing `str` values corresponding to the id of the clusters.
        >>> clusters = ['cluster_1', 'cluster_2']

        Output
        ---
        Return a `dict` containing as keys the names of the clusters and as values the corresponding
        total entropy.
        '''

        # Initialize a dictionary {cluster : tot_entropy}.
        cluster_entropy = {cluster: 0 for cluster in clusters}

        # Iterate over the clients.
        for client in client_entropy.keys():
            
            # Access the cluster of the client.                   # Sum the entropy of that client.
            cluster_entropy[client_entropy[client]['cluster']] += client_entropy[client]['entropy']

        return cluster_entropy



    def aggregate(self, client_entropy: dict, client_num_samples, clusters, sigma, client_model):
        '''
        Make an aggregation taking into account both the entropy and the number of images as weights.

        Parameters
        ---
        client_weight : dict
            `dict` object containing the relative weight of each client depending on the entropy of its cluster;
            >>> client_weight = {'client_1' = 0.7,
            >>>                  'client_2' = 0.3}
        
        client_entropy : dict
            `dict` object containing the clients as keys and as a values another `dict`
            containing the value of the entropy and the corresponding cluster;
            >>> client_entropy = {'client_1' : {'entropy' : 0.44,
            >>>                                 'cluster' : 'cluster_1'}}
        
        cluster_card : dict
            `dict` object containing the clusters as keys and as values their cardinalities.
            >>> cluster_card = {'cluster_1' : 5,
            >>>                 'cluster_2' : 7}

        TODO
        client_model : ???

        entropy_weight : float, default = 0.5
                Relative weight of the entropy with respect to the number of images per client.
                They must sum to 1;

        '''
        #m_tot_samples = 0
        
        img_weight_dict = self.get_weights_image(client_num_samples)
        entropy_weight_dict = self.get_weights_entropy(client_entropy, clusters) 


        base = OrderedDict()

        print("\nsigma", sigma)
        for client in entropy_weight_dict.keys():
            tot_weight_client = ((1 - sigma) * img_weight_dict[client] + sigma * entropy_weight_dict[client])
            
            print(f"Client {client.name}:")
            print(f"\t-num_img:{client_num_samples[client.name]}")
            print(f"\t-client_entropy:{client_entropy[client.name]}")
            print(f"\t-tot_weight: {tot_weight_client}\n\n")
            
            for key, value in client_model.items():
                if key in base:
                    base[key] += tot_weight_client * value.type(torch.FloatTensor)
                else:
                    base[key] = tot_weight_client * value.type(torch.FloatTensor)
        
        averaged_sol_n = copy.deepcopy(self.model_params_dict)

        for key, value in base.items():
            averaged_sol_n[key] = value.to('cuda')

        return averaged_sol_n #è un model_state_dict
