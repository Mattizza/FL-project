
import numpy as np

class ClientSelector():


    def __init__(self, train_clients, alpha: float = 0.0, beta: float = 0.4, llambda: float = 1.0):
        '''
        Initialize a ClientSelection object that can handle both FedAvg and FedCCE. It manages all
        the client selection pipeline. The gamma parameter will be the complement to 1 of alpha and
        beta.

        Parameters
        ---
        alpha : float, default = 0.2
            Weight related to the number of samples of the client. It must be comprised in `[0, 0.5]`;
        
        beta : float, default = 0.4
            Weight related to the entropy of the client. It must be comprised in `[0, 0.5]`;
        
        llambda : float, default = 1.0
            Lambda parameter in the sigmoid function.
        '''
        self.train_clients = train_clients

        self.alpha = alpha
        self.beta = beta
        self.gamma = 1 - self.alpha - self.beta     # Complement to 1.
        self.llambda = llambda



    
    def set_params(self, alpha: float, beta: float, llambda: float) -> None:
        '''
        Set the parameters related to the relative importance of each weight and the sigmoid.

        Parameters
        ---
        alpha : float
            Weight related to the number of samples of the client. It must be comprised in `[0, 0.5]`;
        
        beta : float
            Weight realted to the entropy of the client. It must be comprised in `[0, 0.5]`;
        
        llambda : float
            Lambda parameter in the sigmoid function.
        '''

        self.alpha = alpha
        self.beta = beta
        self.gamma = 1 - self.alpha - self.beta
        self.llambda = llambda


    def _get_weights_image(self, client_num_samples: dict) -> dict:
        '''
        Compute the weights with respect to the number of samples per client.

        Parameters
        ---
        client_num_samples : dict
            `dict` object containing the clients as keys and as a values the number
            of samples it contains.
        
        Output
        ---
        Returns a `dict` containing the weights of the clients with respect to the number
        of samples: `{clients : samples_weights}`.
        '''
        
        # Compute the total number of images in the round.
        tot_img = sum(list(client_num_samples.values()))
        client_img_weight = {c: 0 for c in client_num_samples.keys()}   # Initialize a dictionary {client : sample_weight}.

        # Iterate over all the clients.
        for c in client_num_samples.keys():
            
            # Sample weight for the client computed as a proportion with the total.
            client_img_weight[c] = client_num_samples[c] / tot_img

        return client_img_weight


    def _compute_tot(self, client_entropy: dict, weight: str = 'entropy') -> float:
        '''
        Computes the total entropy/loss at the end of the last epoch.

        Parameters
        ---
        client_entropy : dict
            `dict` object containing the clients as keys and as a values another `dict`
            containing the value of the entropy and the loss;
        >>> client_entropy = {'client_1' : {'entropy' : 0.44,
        >>>                                 'loss'    : 0.60}}

        weight : str, default = 'entropy'
            It specifies whether to compute the total entropy or loss. The only two
            accepted inputs are 'entropy' or 'loss'.
        
        Output
        ---
        Return a `float` value representing the total entropy/loss.
        '''

        tot = 0

        if weight == 'entropy':
            
            # Iterate over the clients.
            for client in client_entropy.keys():
                
                # Sum the entropies of the clients.
                tot += client_entropy[client][weight]
        
        else:

            # Iterate over the clients.
            for client in client_entropy.keys():
                
                # Sum the sigmoid of losses of the clients.
                tot += self._get_sigmoid(client_entropy[client][weight], self.llambda)

        return tot
    

    def _get_sigmoid(self, x: np.dtype, llambda: float = 1) -> np.dtype:
        '''
        Given an array `x` it computes its sigmoid value. Varying `llambda` determines the intensity
        of the jump around the zero value.

        Parameters
        ---
        x : np.array
            Input argument of the sigmoid;
        
        llambda : float, default = 1
            Parameters that determines the shape of the sigmoid. Higher values result in higher jumps.
        
        Output
        ---
        Returns the value of the sigmoid for each value in `x`.            
        '''

        return 1 / (1 + np.exp(-llambda * x))
    

    def _get_weights_entropy_loss(self, client_entropy: dict, weight = 'entropy') -> dict:
        '''
        Compute the weights with respect to the magnitude of the entropy/loss 
        over the total.

        Parameters
        ---
        client_entropy : dict
            `dict` object containing the clients as keys and as a values another `dict`
            containing the value of the entropy and the loss;
        >>> client_entropy = {'client_1' : {'entropy' : 0.44,
        >>>                                 'loss'    : 0.60}}
        
        weight : str, default = 'entropy'
            It specifies whether to compute the total entropy or loss. The only two
            accepted inputs are 'entropy' or 'loss'.

        Output
        ---
        Returns a `dict` containing the weights of the clients with respect to the
        entropy: `{clients : weights}` 
        '''

        clients_id = client_entropy.keys()                      # Store the id of the clients.
        client_weight = {client: 0 for client in clients_id}    # Initialize a dictionary {client : weight}.
        
        if weight == 'entropy':

            # Compute the total entropy.
            tot_entropy = self._compute_tot(client_entropy, weight)

            # Iterate over the clients, compute and assign the weight corresponding to the relative entropy.
            for client in client_entropy.keys():
        
                # Access the client     # Access the entropy and get the relative weight.
                client_weight[client] = client_entropy[client][weight] / tot_entropy

            return client_weight

        else:

            # Compute the sigmoid of the losses for each cluster.
            tot_loss = self._compute_tot(client_entropy, weight)

            # Iterate over the clients, compute and assign the weight corresponding to the relative sigmoid of the loss.
            for client in client_entropy.keys():
        
                # Access the client     # Access the loss and get the relative weight.
                client_weight[client] = self._get_sigmoid(client_entropy[client][weight], self.llambda) / tot_loss

            return client_weight


    def compute_probs(self, client_entropy, client_num_samples) -> dict:
        '''
        Compute, for each client, the probability to be picked in the next round considering
        its number of samples, its entropy and its loss.

        Parameters
        ---
        client_entropy : dict
            `dict` object containing the clients as keys and as a values another `dict`
            containing the value of the entropy and the loss;
        >>> client_entropy = {'client_1' : {'entropy' : 0.44,
        >>>                                 'loss'    : 0.60}}

        client_num_samples : dict
            `dict` object containing the clients as keys and as a values the number
            of samples it contains.
        
        Output
        ---
        A `dict` containing, for each client, the probability to be picked in the next round.
        '''

        # Get all the weights for each client.
        client_img_weight_dict = self._get_weights_image(client_num_samples)
        client_entropy_weight_dict = self._get_weights_entropy_loss(client_entropy, 'entropy')
        client_loss_weight_dict = self._get_weights_entropy_loss(client_entropy, 'loss')

        client_probs = {client: 0 for client in client_entropy.keys()}


        for client in client_entropy.keys():

            # Compute the probabilities.
            client_probs[client] += (self.alpha * client_img_weight_dict[client] + \
                                    self.beta  * client_entropy_weight_dict[client] + \
                                    self.gamma * client_loss_weight_dict[client]) / (1/len(client_entropy.keys())) * (1 / len(self.train_clients))
            
        print("\nclient_probs", client_probs)
        dict_p = {client.name: 1/len(self.train_clients) for client in self.train_clients}

        for client, prob in client_probs.items():
            dict_p[client] = prob

        print("\nNew dict_p", dict_p)
        
        p_return = []
        for client in self.train_clients:
            p_return.append(dict_p[client.name])
        
        return p_return
    