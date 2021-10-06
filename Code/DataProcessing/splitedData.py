import random
import numpy as np

# local imports
from DataProcessing.user import *


class SplitedDataset(Dataset):
    """
    Class that splits training and test set into more manageable datasets.
    """
    def __init__(self, images_dict, images_df, cuda=False):
        """
        Initializes class SplitedDataset.
        Parameters
        ----------
        images_dict : dict
            Dictionary whose keys are images indices and values are feature 28x28 images (arrays).
        images_df : Pandas Dataframe
            Dataframe whose columns are image indices (refer to images_dict), corresponding labels and writers.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        super().__init__()
        self.images_dict = images_dict
        self.images_df = images_df
        self.N = len(self)
        self.params = None
        self.cuda = cuda
        self.transform = Compose([Normalize(), ToTensor(self.cuda)])

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if isinstance(item, int): # a single item selected
            image = np.transpose(self.images_dict[self.images_df.images[item]].reshape((28, 28)))
        else:
            images = []

            for im in [self.images_dict.get(key) for key in self.images_df.images[item].values]:
                images += [np.transpose(im.reshape((28, 28)))]
            image = np.array(images)
        label = self.images_df['labels'][item].values
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample

    def C_random_select(self, C=1):
        """
        Samples a proportion C of the users randomly and only for the train set.
        The test set will be used entirely when evaluating the model, thus there is no need to sample a portion of it.
        Parameters
        ----------
        C : float
            Proportion of the train set to be sampled. Default is 1.
            0<C<1 : C=0 corresponds to training on only one user. C=1 corresponds to training on all users.
        Returns
        -------
        selected_users : list
            List of indexes of C randomly sampled users.
        """
        assert not self.images_df.empty, "Dataframes empty. First preprocess data"

        total_nb_users = len(self.images_df.writers.unique())
        K = max(round(C * total_nb_users), 1)
        # Select proportion C of clients randomly
        selected_users = random.sample(list(self.images_df.writers.unique()), k=K)

        return selected_users
