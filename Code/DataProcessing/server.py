import torch


class Server:
    """
    Class that represent the central server in federated learning.
    """
    def __init__(self, net, cuda=False):
        """
        Initializes class Server.
        Parameters
        ----------
        net : class Network
            Network to be used by the user.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        self.net = net
        self.N = 0
        self.cuda = cuda
        if self.cuda:
            self.net = self.net.to('cuda')
        self.average_weight = self.net.state_dict()
        self.received_weights = {}
        self.losses_test_rounds = []
        self.accuracy_test_rounds = []
        self.cuda = cuda

    def send_to_users(self, users):
        """
        Sends Server's network parameters to users.
        Parameters
        ----------
        users : list
            List of class User instances to which send the Server's network parameters.
        """
        for user in users:
            user.net.load_state_dict(self.net.state_dict())

    def receive_from_user(self, params, nk):
        """
        Server accumulates parameters received by a user.
        Parameters
        ----------
        params : dict
            Dictionary whose keys are the user's layers names and values their corresponding parameters.
        nk : int
            Number of samples of user.
        """
        # If params is empty, no user has sent its weights.
        if not self.received_weights:
            for key in params.keys():
                self.received_weights[key] = [nk*params[key]]
        # If params already contains values, some users have already sent their weights.
        else:
            for key in params:
                self.received_weights[key] += [nk*params[key]]
        self.N += nk

    def update_average_weights(self):
        """
        Server averages its received parameters.
        """
        for layer in self.received_weights.keys():
            self.average_weight[layer] = self.net.state_dict()[layer] - \
                                         (1/self.N)*torch.sum(torch.stack(self.received_weights[layer]),
                                                              0, keepdim=True).squeeze(0)
        # Update server net
        self.net.load_state_dict(self.average_weight)
        # Reset received weights to avoid accumulation between rounds
        self.received_weights = {}
        self.N = 0




