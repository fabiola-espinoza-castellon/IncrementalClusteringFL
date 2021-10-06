import pandas as pd
from scipy.io import loadmat

# local imports
from DataProcessing.splitedData import *


class EntireDataset(Dataset):
    """
    Class that handles MNIST/EMNIST data for federated learning.
    """

    def __init__(self, root_dir, cuda=False):
        """
        Initialize EntireDataset class.
        Parameters
        ----------
        root_dir : str
            Directory path containing .mat file of MNIST or EMNIST data.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        super().__init__()
        self.root_dir = root_dir
        self.labels2chr = None
        self.selected_users = None
        self.users = {}
        self.train_data = None
        self.test_data = None
        self.cuda = cuda

    def preprocess_data(self, K, select_type='top'):
        """
        Sets attributes train_data and test_data to SplitedDataset instances that contain the train data and the test
         data respectively. This data is reduced to K users (selected from the train set).
         Parameters
         ----------
         K : int
            Number of users to be selected.
         select_type: string
            Indicating the way users will be selected. 3 possible entries :
            'random' selects K users randomly.
            'top' (default) selects K users with most data.
            'bottom' selects K users with less data.
        """
        all_data = loadmat(self.root_dir)['dataset']
        assert isinstance(select_type, str), "Choose how to select users. Only valid format are 'random', " \
                                             "'top' or 'bottom'."
        if all_data[0][0][2].size:
            self.labels2chr = dict(zip(all_data[0][0][2][:, 0], all_data[0][0][2][:, 1]))
        else:
            raise Exception('Labels/characters dictionary not found')

        # First : load train data and select users from this set
        dic, df = load_data(all_data[0][0][0])
        if select_type == 'random':
            self.selected_users = random.sample(list(df.writers.unique()), k=K)
        elif select_type == 'top':
            # Reset index of writers from highest to lowest
            sorted_users = df.groupby('writers').count().sort_values('images', ascending=False).index.values
            df.writers = df.writers.replace(sorted_users, range(len(sorted_users)))
            self.selected_users = range(K)
        elif select_type == 'bottom':
            sorted_users = df.groupby('writers').count().sort_values('images').index.values
            df.writers = df.writers.replace(sorted_users, range(len(sorted_users)))
            self.selected_users = range(K)

        df = df[df.writers.isin(self.selected_users)].reset_index()
        self.train_data = SplitedDataset(dic, df, self.cuda)

        # Second : load test data and reduce clients
        dic, df = load_data(all_data[0][0][1])
        df.writers = df.writers.replace(sorted_users, range(len(sorted_users)))
        df = df[df.writers.isin(self.selected_users)].reset_index()
        self.test_data = SplitedDataset(dic, df, self.cuda)

    def create_users(self, user_net, user_params):
        """
        Returns and sets attribute users to a list of User instances based on the user values in the data. All users
        will have the same network architecture user_net.
        Parameters
        ----------
        user_net : class Network
            Defines the network of users.
        user_params : dict
            Dictionary containing as local parameters.
        Returns
        -------
        List of instances of class User.
        """
        for user in self.train_data.images_df.writers.unique():
            keys_train = self.train_data.images_df[self.train_data.images_df.writers == user].index
            keys_test = self.test_data.images_df[self.test_data.images_df.writers == user].index

            images_train = self.train_data[keys_train]['image']
            images_test = self.test_data[keys_test]['image']

            labels_train = self.train_data[keys_train]['label']
            labels_test = self.test_data[keys_test]['label']

            self.users[user] = User(user, labels_train, labels_test, images_train, images_test, user_net, user_params, cuda=self.cuda)
        return self.users

    def filter_by_users(self, users_id):
        """
        Filters users list by ids.
        Parameters
        ----------
        users_id : list
            Ids of users to be kept.
        """
        self.train_data.images_df = self.train_data.images_df[self.train_data.images_df.writers.isin(users_id)]
        self.test_data.images_df = self.test_data.images_df[self.test_data.images_df.writers.isin(users_id)]
        self.users = {k: v for k, v in self.users.items() if k in users_id}


def load_data(data):
    """
    Loads data previously read from .mat file.
    Parameters
    ----------
    data : numpy array
        Array of shape

    Returns
    -------
    images_dict : dict
        Dictionary which keys are indices and values are feature 28x28 images (arrays).
    images_df : Pandas Dataframe
        Dataframe whose columns are image indices (refer to images_dict), corresponding labels and writers.
    """
    images_dict = dict(zip(np.arange(0, np.shape(data[0][0][0])[0]), data[0][0][0]))
    a = np.column_stack((list(images_dict.keys()), data[0][0][1]))
    a = np.column_stack((a, data[0][0][2]))
    columns = ['images', 'labels', 'writers']
    images_df = pd.DataFrame(a, columns=columns)
    return images_dict, images_df
