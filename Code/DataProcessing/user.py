from torch.utils.data import Dataset
from torchvision.transforms import Compose

# local imports
from DataProcessing.transforms import *


class User(Dataset):
    """
    Class that represent a user in federated learning.
    """
    def __init__(self, idx, train_labels, test_labels, train_images, test_images, net, local_hyperparams,
                 clusters_train=None, clusters_test=None, cuda=False):
        """
        Initializes class User
        Parameters
        ----------
        idx : int
            User index.
        train_labels : torch Tensor
            Tensor of shape (number of samples) with values of train labels.
        test_labels : torch Tensor
            Tensor of shape (number of samples) with values of test labels.
        train_images : torch Tensor
            Tensor of shape (number of samples, 28, 28) with train images.
        test_images : torch Tensor
            Tensor of shape (number of samples, 28, 28) with test images.
        net : class Network
            Defines user's network.
        local_hyperparams : dict
            Dictionary whose values define user's locals hyperparameters specifically as 'lr' (learning rate) and
            'criterion' (loss function).
        clusters_train : numpy array
            Array of shape (number of samples,) with the clusters corresponding to each train samples data distribution.
        clusters_test : numpy array
            Array of shape (number of samples,) with the clusters corresponding to each test samples data distribution.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        self.idx = idx
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_images = train_images
        self.test_images = test_images
        self.clusters_train = clusters_train
        self.clusters_test = clusters_test
        self.cuda = cuda
        self.net = net(local_hyperparams['inputChannel'], local_hyperparams['outputSize'])
        if self.cuda:
            self.net = self.net.to('cuda')
        self.lr = local_hyperparams['lr']
        self.criterion = local_hyperparams['criterion']
        self.transform = Compose([Normalize(), ToTensor(self.cuda)])
        self.split = 'train'
        self.params = None
        self.nk = len(self)
        self.round_losses_test = []
        self.round_accuracy = []
        self.found_cluster = None
        self.last_round = 0

    def __len__(self):
        if self.split == 'train':
            label_set = self.train_labels
        elif self.split == 'test':
            label_set = self.test_labels
        return len(label_set)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.split == 'train':
            image_set = self.train_images
            label_set = self.train_labels
        elif self.split == 'test':
            image_set = self.test_images
            label_set = self.test_labels

        image = image_set[item]
        label = label_set[item]
        sample = {'image': image, 'label': label}

        return sample

    def train_set(self):
        self.split = 'train'

    def test_set(self):
        self.split = 'test'