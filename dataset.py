import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import glob
import random
from PIL import Image
import os
T = transforms

class CelebADataset(torch.utils.data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, mode='train'):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the image filenames."""
        img_paths = glob.glob(self.image_dir + '/*.jpg')
        self.train_dataset = img_paths

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(1)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(image_dir, crop_size, image_size, batch_size, mode, dataset_name, num_workers=8):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    if 'celeba' not in dataset_name:
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))
    transform = T.Compose(transform)

    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=image_dir, transform=transform, download=True)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=image_dir, transform=transform, download=True)
    elif dataset_name == 'imagenet':
        dataset = torchvision.datasets.ImageFolder(root=image_dir ,transform=transform)
    else:
        dataset = CelebADataset(image_dir, transform, mode)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader