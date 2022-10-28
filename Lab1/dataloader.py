from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas
from glob import glob
import re


class loader(data.Dataset):
    def __init__(self, mode = 'train', transform = None):
        
        self.mode = mode
        print(f'Mode "{self.mode}"')
        if mode == 'train':
            self.data = pandas.read_csv('train.csv')
        elif mode == 'val':
            self.data = pandas.read_csv('val.csv')
        else:
            self.data = glob('test/*')

        self.transform = transform


    def __len__(self):
        if self.mode == 'test':
            return len(self.data)
        else:
            return self.data.shape[0]

    def __getitem__(self, index):

        preprocess = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0.05)])

        try:
            img = Image.open( self.mode  + '/'+ self.data['names'][index])
            label = self.data['label'][index]
        except:
            img = Image.open( self.data[index])
            label = self.data[index]
            label = re.sub('.*/', '', label)

        if self.transform != None:
            img = preprocess(img)

        img = transforms.ToTensor()(img)


        return img, label


if __name__ == '__main__':
    # test = loader()
    print(glob('test/*'))