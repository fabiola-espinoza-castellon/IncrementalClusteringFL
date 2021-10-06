import torch


class Normalize(object):
    """
    Normalizes images by dividing pixels by 255
    To do : Simple for now, see if need to standarize (mean + std)
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image / 255
        return {'image': image, 'label': label}


class ToTensor(object):
    """
    Converts ndarrays in sample to Tensors.
    """
    def __init__(self, cuda):
        self.cuda = cuda

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # no need to swap color axis because grayscale image
        # transform to type float because net weights are in float by default
        if self.cuda and torch.cuda.is_available():
            return {'image': torch.from_numpy(image).type(torch.float).to('cuda'),
                    'label': torch.tensor(label).type(torch.float).to('cuda')}
        elif self.cuda and not torch.cuda.is_available():
            print('Cuda option activated but not running on gpu')
        else:
            return {'image': torch.from_numpy(image).type(torch.float),
                    'label': torch.tensor(label).type(torch.float)}