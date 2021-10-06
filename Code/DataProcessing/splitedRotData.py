from torchvision.transforms import RandomRotation

# local imports
from DataProcessing.splitedData import *


class SplitedRotDataset(SplitedDataset):
    """
    Class that splits training and test set into more manageable datasets for the non-IID case image rotation.
    """

    def __init__(self, images_dict, images_df, cuda=False):
        """
        Initializes class SplitedRotDataset.
        Parameters
        ----------
        images_dict : dict
            Dictionary whose keys are images indices and values are feature 28x28 images (arrays).
        images_df : Pandas Dataframe
            Dataframe whose columns are image indices (refer to images_dict), corresponding labels and writers.
        cuda : boolean
            Boolean indicating if GPU should be used or not. Default is False.
        """
        super().__init__(images_dict, images_df, cuda=False)
        self.images_dict = images_dict
        self.images_df = images_df
        self.N = len(self)
        self.params = None
        self.cuda = cuda
        self.transform = Compose([Normalize()])

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if isinstance(item, int): # a single item selected
            image = torch.Tensor(np.transpose(self.images_dict[self.images_df.images[item]].reshape((28, 28)))).unsqueeze(0)
            rot = self.images_df.rotation[item]
            image = RandomRotation([rot, rot])(image)
            label = torch.tensor(self.images_df['labels'][item]).type(torch.float)
        else:
            images = []
            for im, rot in zip([self.images_dict.get(key) for key in self.images_df.images[item].values],
                               self.images_df.rotation[item].values):
                images += [RandomRotation([rot, rot])(torch.Tensor(np.transpose(im.reshape((28, 28)))).unsqueeze(0))]
            image = torch.vstack(images)
            label = torch.tensor(self.images_df['labels'][item].values).type(torch.float)
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)

        if self.cuda and torch.cuda.is_available():
            return {k: v.to('cuda') for k, v in sample.items()}

        else:
            return sample
