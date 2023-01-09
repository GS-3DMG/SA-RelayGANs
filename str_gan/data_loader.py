import sys
sys.path.append("/mnt/data/czs/pix2pix/pix2pix")
import torch
import torchvision.datasets as dsets
from torchvision import transforms

from dataset import SimpleDatasetFromLoader


class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # options.append(transforms.Grayscale)
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_bice(self):
        transforms = self.transform(False, False, True, False)
        dataset = SimpleDatasetFromLoader("data/bice2", train=True, transform=transforms)
        return dataset

    def load_lena(self):
        transforms = self.transform(False, False, True, False)
        dataset = SimpleDatasetFromLoader("/mnt/data/czs/pix2pix/pix2pix/dataset/lena/train/b", train=True, transform=transforms)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'bice':
            dataset = self.load_bice()
        elif self.dataset == 'lena':
            dataset = self.load_lena()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              num_workers=8,
                                              drop_last=True)
        # loader = SimpleDatasetFromLoader("data/bice2", train=True, transform=transforms)
        return loader

