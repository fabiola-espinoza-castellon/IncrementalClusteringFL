from DataProcessing.entireData import *


class LabelSwapData(EntireDataset):
    """"
    Class that handles MNIST/EMNIST data for federated learning in the non-IDD case label swap (concept shift).
    """

    def __init__(self, root_dir, swaps, K, cuda=False):
        """
        Initialize LabelSwapData class.
        Parameters
        ----------
        root_dir : str
            Directory path containing .mat file of MNIST or EMNIST data.
        swaps : dict
            Nested dictionary whose keys are clusters ids and values the corresponding label swaps (dict).
        K : int
            Number of users to be created.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        super().__init__(root_dir, cuda)
        self.nb_clusters = len(swaps)
        self.swaps = swaps
        self.K = K

    def balance_users(self, df):
        # We must use a balanced dataset ie same number of samples per class
        # Now, we want the same number of labels per client
        df = df.sort_values('labels')
        users = np.arange(self.K)
        m = int(df.groupby('labels').count().iloc[0][0] / self.K)
        users_label = []
        for u in users:
            users_label += [u] * m
        users_label = users_label * len(df.labels.unique())
        df.writers = users_label
        return df

    def label_swap(self, df):
        """
        swap: (nested_dict) keys are cluster number and values are 2 length dicts whose keys are labels to be swaped
        Ex: {0:{1:2, 2:1}, 1:{3:4,4:3}} will swap 1 to 2 and 2 to 1 for cluster 0 and 3 to 4 and 4 to 3 for cluster 1
        """
        df = df.sort_values('writers')
        users = df.writers.unique()
        len_clusters = len(users) // self.nb_clusters  # users per cluster
        all_labels = pd.Series([], dtype='float64')
        clusters = []
        for c in np.arange(self.nb_clusters):
            labels = df[df.writers.isin(np.arange(c * len_clusters, (c + 1) * len_clusters))].labels
            labels = labels.replace(self.swaps[c])
            all_labels = pd.concat([all_labels, labels])
            clusters += [c] * len(labels)

        df.labels = all_labels
        df['clusters'] = clusters

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
        df_train, df_test = self.label_swap(df_train), self.label_swap(df_test)

        self.train_data = SplitedDataset(dic_train, df_train, self.cuda)
        self.test_data = SplitedDataset(dic_test, df_test, self.cuda)

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


