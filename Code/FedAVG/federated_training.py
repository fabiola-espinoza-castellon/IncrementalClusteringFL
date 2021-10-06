from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import networkx as nx
import community as community_louvain

# local imports
from Metrics.metrics import *
from DataProcessing.server import *
from Test.test_federated_training import *
from Networks.CNN import *
from Utils.save_files import *

test = TestFederatedTraining()


class FedAVG:

    def __init__(self, server_net, users_net, data, global_hyperparams, local_hyperparams, preload=False, cuda=False,
                 resultsPath=None):
        """
        Initialize FedAVG class.
        Parameters
        ----------
        server_net : class Network
            Defines the network of the server.
        users_net : class Network
            Defines users' network, same class as the server's.
        data : class Dataset
            Defines the data to be used. Generally partition into the wanted non-IID case.
        global_hyperparams : dict
            Defines global parameters.
        local_hyperparams : dict
            Defines local parameters.
        preload : boolean (default: False), str or Network class
            If different from False, preload is the path (str) to the .pth file of the server's network or the Network
            class itself.
        cuda : boolean (default: False)
            If True computation will be done in GPU.
        resultsPath : str (default: None)
            Path where to save results.
        """
        self.users_net = users_net
        self.data = data
        self.local_hyperparams = local_hyperparams
        self.global_hyperparams = global_hyperparams
        self.users = None
        self.server = None
        self.rounds = 0
        self.preload = preload
        if self.preload:
            self.server_net = server_net
        else:
            self.server_net = server_net(local_hyperparams['inputChannel'], local_hyperparams['outputSize'])
        self.similarity_matrix = None
        self.representation_matrix = None
        self.clusters_nets = {}
        self.unclustered = None
        self.cuda = cuda
        self.resultsPath = resultsPath

    def initialize(self):
        """
        Initializes server network, similarity and representation matrix and creates users.
        """
        if self.preload:
            net = CNNetwork(1, self.local_hyperparams['outputSize'])
            if isinstance(self.server_net, str):
                net.load_state_dict(torch.load(self.server_net))
            else:
                net.load_state_dict(self.server_net.state_dict())
            self.server_net = net
            self.server_net.eval()
            self.server = Server(self.server_net, cuda=self.cuda)
        else:
            self.server = Server(self.server_net, cuda=self.cuda)

        self.users = self.data.create_users(self.users_net, self.local_hyperparams)

        self.similarity_matrix = torch.eye(len(self.users)) * 2  # because distance cosine <2
        self.representation_matrix = torch.Tensor(
            [[np.nan] * sum(p.numel() for p in self.server.net.parameters())] * len(self.users))

    def client_update(self, user):
        """
        Performs local optimization for a client.
        Parameters
        ----------
        user : class User
            User who will perform local update.
        Returns
        -------
        update : dict
            Dictionary containing user's network's parameters updated.

        """
        test.test_same_weights_net(self.server.net, user.net)
        user.train_set()

        for epoch in range(1, self.local_hyperparams['E'] + 1):
            dataloader_train = DataLoader(user, batch_size=self.local_hyperparams['B'], shuffle=True, num_workers=0)
            for batch_idx, data in enumerate(dataloader_train):
                user.optimizer = torch.optim.SGD(user.net.parameters(), lr=user.lr)
                user.optimizer.zero_grad()
                user_net_out = user.net(data['image'].unsqueeze(1).float())
                loss = user.criterion(user_net_out, data['label'].long())
                loss.backward()
                user.optimizer.step()
        update = {}
        for key in user.net.state_dict().keys():
            update[key] = self.server.net.state_dict()[key] - user.net.state_dict()[key]
        return update

    def start_fedavg_training(self, send_server=True, online_cluster=False, save_accuracies=True,
                              filename_cluster='FedAvg'):
        """
        Performs federated training for users with the option to perform incremental clustering.
        Parameters
        ----------
        send_server : boolean
            Defines if updates are sent to server. Default is true.
        online_cluster : boolean
            True if incremental clustering is to be applied (update representation matrix and similarity matrix).
            Default is False.
        save_accuracies : boolean
            Accuracies evaluated on server will be saved as an .npy array of shape (number of rounds). Default is True.
        filename_cluster : str
            Filename to be used to save results. Default is 'FedAvg'.
        Returns
        -------
        server.accuracy_test_rounds : list
            List containing servers accuracies trough rounds evaluated on test images of all users managed by server.
        """
        self.initialize()
        for r in range(self.global_hyperparams['rounds']):
            # Evaluate model right after sending new weights
            self.server.send_to_users(self.users.values())
            self.test_server()
            self.test_users()
            # Select C users
            random_users = self.data.train_data.C_random_select(C=self.global_hyperparams['C'])
            C_users = [self.users[id_] for id_ in random_users]
            N = 0  # total number of samples
            for user in C_users:
                user.last_round = self.rounds + 1
                params = self.client_update(user)
                N += user.nk
                if online_cluster:
                    self.representation_matrix[user.idx] = torch.hstack([torch.flatten(p) for p in params.values()])

                if send_server:
                    self.server.receive_from_user(params, user.nk)

            if online_cluster:
                update_similarity_matrix(self.representation_matrix, self.similarity_matrix,
                                              metric='cosine')

            if send_server:
                self.rounds += 1
                self.server.update_average_weights()
                test.test_well_updated_server(self.server, N)
        if save_accuracies:
            save_arrays(self.server.accuracy_test_rounds, "accuracyServer_{}".format(filename_cluster),
                        self.resultsPath)
        return self.server.accuracy_test_rounds

    def clustering_training(self, rounds=1, filename='cluster', **kwargs):
        """
        Performs FedAvg per cluster independently.
        Parameters
        ----------
        clusters_list_filename : str OR dict
            Path to JSON file containing partition dict OR partition dict.
        rounds : int
            Number of rounds per cluster. Default 1.
        filename : str
            Save results with filename. Default 'cluster'.
        kwargs :
            start_fedavg_training parameters save_server_accuracies (default True), last_evaluate (default False),
            send_server (default True).
        """
        clusters_users = cluster_louvain(self)
        save_cache_local_mem(self.data, 'data', self.resultsPath)
        save_cache_local_mem(self.server, 'server', self.resultsPath)

        cluster_params = self.global_hyperparams
        cluster_params['rounds'] = rounds
        for cluster in clusters_users:
            data = load_cache_local_mem('data', self.resultsPath)
            data.filter_by_users(clusters_users[cluster])
            server = load_cache_local_mem('server', self.resultsPath)

            cluster_fedavg = FedAVG(server.net, self.users_net, data, cluster_params,
                                            self.local_hyperparams, preload=True, resultsPath=self.resultsPath)

            cluster_fedavg.start_fedavg_training(send_server=kwargs.get('send_server', True), online_cluster=False,
                                        save_accuracies=kwargs.get('save_server_accuracies', True), 
                                        filename_cluster='{}{}'.format(filename, cluster))
            # Keep clusters networks
            self.clusters_nets[cluster] = CNNetwork(1, self.local_hyperparams['outputSize'])
            self.clusters_nets[cluster].load_state_dict(cluster_fedavg.server.net.state_dict())
                                        
    def choose_unclustered(self):
        """
        If some users were not sampled during FedAvg training, thus not clustered, we can still assign them to a cluster
        by evaluating their test images on each cluster's network. The user will be assigned to the cluster 
        corresponding to the highest accuracy.
        """
        for user in self.unclustered:
            accs = {}
            for c in self.clusters_nets:
                self.users[user].test_set()
                user_data = self.users[user][:]
                with torch.no_grad():
                    accs[c] = get_accuracy(user_data['image'].unsqueeze(1), user_data['label'],
                                           self.clusters_nets[c])
            self.users[user].found_cluster = max(accs, key=accs.get)
            print('User {} assigned to cluster {}'.format(user, self.users[user].found_cluster))

    ####################################################################################################################
    # Utility FedAvg functions
    ####################################################################################################################

    def test_users(self, selected_users=None):
        """
        Test users' federated model individually. Values will be stored on users' attribute 'round_accuracy' and
        'round_losses_test'.
        Parameters
        ----------
        selected_users : list OR None
            List of users for which evaluation is to be performed. Default is None. In this case, all users managed by
            server will be evaluated.
        """
        if selected_users:
            users = selected_users
        else:
            users = self.users.values()

        for user in users:
            user.test_set()
            user_data = user[:]
            with torch.no_grad():
                user.round_accuracy += [get_accuracy(user_data['image'].unsqueeze(1), user_data['label'], user.net)]
                user.round_losses_test += [user.criterion(user.net(user_data['image'].unsqueeze(1)),
                                                          user_data['label'].long()).data.item()]

    def test_server(self):
        """
         Test federated model on all users managed by server. Values will be stored on server's attribute
         'accuracy_test_rounds' and 'losses_test_rounds'.
         """
        with torch.no_grad():
            test_data = self.data.test_data[:]
            self.server.accuracy_test_rounds += [get_accuracy(test_data['image'].unsqueeze(1), test_data['label'],
                                                              self.server.net)]

            self.server.losses_test_rounds += [self.global_hyperparams['criterion'](
                self.server.net(test_data['image'].unsqueeze(1)), test_data['label'].long())]


########################################################################################################################
# Utility FedAvg functions
########################################################################################################################

def cluster_louvain(class_FedAVG):
    """
    Clusters users of class_FedAVG wiht the Louvain method.
    Parameters
    ----------
    class_FedAVG : class FedAVG
        Instance of FedAVG that already contains users and has been trained.

    Returns
    -------
    clusters_identities : dict
        Dictionary whose keys are clusters' ids and values are lists of users' ids belonging to the corresponding
        clusters.
    """
    def order_partition(p):
        d = {}
        i = 0
        for k in p.keys():
            val = p[k]
            if val in d.keys():
                p[k] = d[val]
            else:
                d[val] = [i]
                p[k] = [i]
                i += 1
        return p

    # First, filter matrix in order to delete non selected users
    s = class_FedAVG.similarity_matrix - torch.eye(class_FedAVG.similarity_matrix.shape[0]) * 2
    s = pd.DataFrame(s.numpy())
    class_FedAVG.unclustered = s[s == 0].dropna().index
    s_in = s[s != 0].dropna(how='all').dropna(axis=1, how='all')
    s_in = s_in.fillna(2)

    clusters_identities = {}
    graph = nx.from_pandas_adjacency(s_in)
    partition = community_louvain.best_partition(graph)
    partition = order_partition(partition)
    clusters = dict(zip(s_in.index, partition.values()))
    for c in np.unique(list(clusters.values())):
        clusters_identities[c.item()] = list({k: v for k, v in partition.items() if v == c}.keys())
    return clusters_identities


def update_similarity_matrix(rep_matrix, sim_matrix, metric='cosine'):
    """
    Updates similarity matrix by using representation matrix.
    Parameters
    ----------
    rep_matrix : torch Tensor
        Representation matrix containing users' updates from which similarities will be computed.
    sim_matrix : torch Tensor
        Resulting similarity matrix.
    metric : str
        Metric to be used as similarity. Default is 'cosine' and the corresponding similarity between x and y is equal
        to 2 - cosine_distance(x,y) = 1 + cos(x,y).
    """
    s = squareform(pdist(rep_matrix, metric=metric))
    s[s != s] = 0
    s = torch.Tensor(s)
    sim_matrix[s != 0.0] = 2 - torch.Tensor(s[s != 0.0])