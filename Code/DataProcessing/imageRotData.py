from DataProcessing.entireData import *
from DataProcessing.splitedRotData import *


class ImageRotData(EntireDataset):
    """"
    Class that handles MNIST/EMNIST data for federated learning in the non-IDD case image rotation (concept shift).
    """

    def __init__(self, root_dir, rotations, K, cuda=False):
        """
        Initialize ImageRotData class.
        Parameters
        ----------
        root_dir : str
            Directory path containing .mat file of MNIST or EMNIST data.
        rotations : dict
            Dictionary whose keys are clusters ids and values the corresponding image rotations.
        K : int
            Number of users to be created.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        super().__init__(root_dir, cuda)
        self.nb_clusters = len(rotations)
        self.rotations = rotations
        self.K = K

    def balance_users(self, df):
        """
        Balances data in order to have the same number of samples per label for each user.
        Parameters
        ----------
        df : Pandas dataframe
            Dataframe containing columns images indices and labels.
        Returns
        -------
        df : Pandas dataframe
            Dataframe with writers column added indicating the writers indices.
        """
        df = df.sort_values('labels')
        users = np.arange(self.K)
        m = int(df.groupby('labels').count().iloc[0][0] / self.K)  # number of samples per label for each client
        users_label = []
        for u in users:
            users_label += [u] * m
        users_label = users_label * len(df.labels.unique())
        df.writers = users_label
        return df

    def im_rotate(self, df):
        """
        Adds rotation information.
        Parameters
        ----------
        df : Pandas dataframe
            Dataframe containing columns images indices, labels and writers.
        Returns
        -------
        df : Pandas dataframe
            Dataframe with columns added indicating clusters identity and rotation per client.
        """
        df = df.sort_values('writers')
        len_clusters = df.shape[0] // self.nb_clusters  # samples per cluster
        clusters = []
        rotations = []
        for c in np.arange(self.nb_clusters):
            clusters += [c] * len_clusters
            rotations += [self.rotations[c]]*len_clusters

        df['clusters'] = clusters
        df['rotation'] = rotations
        return df

    def preprocess_data(self):
        """
        Sets attributes train_data and test_data to SplitedDataset instances that contain the train data and the test
        data respectively. This data is reduced to K users (selected from the train set).
        """
        all_data = loadmat(self.root_dir)['dataset']
        if all_data[0][0][2].size:
            self.labels2chr = dict(zip(all_data[0][0][2][:, 0], all_data[0][0][2][:, 1]))
        else:
            raise Exception('Labels/characters dictionary not found')

        # Load train and test data
        dic_train, df_train = load_data(all_data[0][0][0])
        dic_test, df_test = load_data(all_data[0][0][1])

        df_train, df_test = self.balance_users(df_train), self.balance_users(df_test)
        df_train, df_test = self.im_rotate(df_train), self.im_rotate(df_test)

        self.train_data = SplitedRotDataset(dic_train, df_train, self.cuda)
        self.test_data = SplitedRotDataset(dic_test, df_test, self.cuda)

    def create_users(self, user_net, user_params):
        """
        Returns and sets attribute users to a list of User instances based on the user values in the data. All users
        will have the same network architecture user_net.
        Parameters
        ----------
        user_net : Network class
            Network that will be applied to users.
        user_params : dict
            Dictionary containing as local parameters.
        Returns
        -------
        List of instances of class User.
        """
        for user in self.train_data.images_df.writers.unique():
            clusters_train = self.train_data.images_df[self.train_data.images_df.writers == user].clusters.values
            clusters_test = self.test_data.images_df[self.test_data.images_df.writers == user].clusters.values

            keys_train = self.train_data.images_df[self.train_data.images_df.writers == user].images.values
            keys_test = self.test_data.images_df[self.test_data.images_df.writers == user].images.values

            labels_train = self.train_data[keys_train]['label']
            labels_test = self.test_data[keys_test]['label']

            images_train = self.train_data[keys_train]['image']
            images_test = self.test_data[keys_test]['image']

            self.users[user] = User(user, labels_train, labels_test, images_train, images_test, user_net, user_params,
                                    clusters_train=clusters_train, clusters_test=clusters_test, cuda=self.cuda)
        return self.users

